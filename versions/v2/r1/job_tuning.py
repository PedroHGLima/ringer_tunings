#!/usr/bin/env python

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")


from Gaugi import load
from Gaugi.messenger.macros import *
from saphyra.utils import model_generator_base
import numpy as np
import argparse
import sys,os


parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-c','--configFile', action='store',
        dest='configFile', required = True,
            help = "The job config file that will be used to configure the job (sort and init).")

parser.add_argument('-v','--volume', action='store',
        dest='volume', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-d','--dataFile', action='store',
        dest='dataFile', required = True, default = None,
            help = "The data/target file used to train the model.")

parser.add_argument('-r','--refFile', action='store',
        dest='refFile', required = True, default = None,
            help = "The reference file.")

parser.add_argument('-t','--type', action='store',
        dest='type', required = True, default = 'fusion',
            help = "The type of derivation (ringer, shower or fusion)")

parser.add_argument('--path_to_rings', action='store',
        dest='path_to_rings', required = False, default = None,
            help = "The path to the fastcalo ringer v10 tuned files.")

parser.add_argument('--path_to_shower', action='store',
        dest='path_to_shower', required = False, default = None,
            help = "The path to the fastcalo v9 shower tuned files.")

parser.add_argument('--path_to_track', action='store',
        dest='path_to_track', required = False, default = None,
            help = "The path to the fast electron v1 track tuned files.")


if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)


args = parser.parse_args()




#
# Get track from file
#
def getPatterns_track( path, cv, sort):

  d = load(path)
  n = d['data'].shape[0]

  feature_names = d['features'].tolist()
  has_track = d['data'][:,feature_names.index('L2Electron_hastrack')]

  # Get only events with L2 electron object to the training phase
  d['target'] = d['target'][has_track==True]
  d['data'] = d['data'][has_track==True]


  data_etOverPt  = d['data'][:, feature_names.index('L2Electron_etOverPt')].reshape((n,1))
  data_deta      = d['data'][:, feature_names.index('L2Electron_trkClusDeta')].reshape((n,1))
  data_dphi      = d['data'][:, feature_names.index('L2Electron_trkClusDphi')].reshape((n,1))
  data  = np.concatenate( (data_etOverPt, data_deta, data_dphi), axis=1)

  target = d['target']
  splits = [(train_index, val_index) for train_index, val_index in cv.split(data_deta,target)]
  x_train = [data [ splits[sort][0]]]
  y_train = target [ splits[sort][0] ]
  x_val = [data [ splits[sort][1]]]
  y_val = target [ splits[sort][1] ]

  return x_train, x_val, y_train, y_val, splits, []





#
# Get shower shapes patterns from file
#
def getPatterns_fusion( path, cv, sort):


  def norm1( data ):
      norms = np.abs( data.sum(axis=1) )
      norms[norms==0] = 1
      return data/norms[:,None]

  d = load(path)
  feature_names = d['features'].tolist()
  has_track = d['data'][:,feature_names.index('L2Electron_hastrack')]

  # Get only events with L2 electron object to the training phase
  d['target'] = d['target'][has_track==True]
  d['data'] = d['data'][has_track==True]


  # How many events?
  n = d['data'].shape[0]

  # extract all rings
  data_rings  = norm1(d['data'][:,1:101])
  # extract all shower shapes
  data_reta   = d['data'][:, feature_names.index('L2Calo_reta')].reshape((n,1)) / 1.0
  data_eratio = d['data'][:, feature_names.index('L2Calo_eratio')].reshape((n,1)) / 1.0
  data_f1     = d['data'][:, feature_names.index('L2Calo_f1')].reshape((n,1)) / 0.6
  data_f3     = d['data'][:, feature_names.index('L2Calo_f3')].reshape((n,1)) / 0.04
  data_weta2  = d['data'][:, feature_names.index('L2Calo_weta2')].reshape((n,1)) / 0.02
  data_wstot  = d['data'][:, feature_names.index('L2Calo_wstot')].reshape((n,1)) / 1.0

  target = d['target']

  # Fix all shower shapes variables
  data_eratio[data_eratio>10.0]=0
  data_eratio[data_eratio>1.]=1.0
  data_wstot[data_wstot<-99]=0


  data_etOverPt  = d['data'][:, feature_names.index('L2Electron_etOverPt')].reshape((n,1))
  data_deta      = d['data'][:, feature_names.index('L2Electron_trkClusDeta')].reshape((n,1))
  data_dphi      = d['data'][:, feature_names.index('L2Electron_trkClusDphi')].reshape((n,1))
  data_track = np.concatenate( (data_etOverPt, data_deta, data_dphi), axis=1)


  # This is mandatory
  splits = [(train_index, val_index) for train_index, val_index in cv.split(data_reta,target)]

  data_shower = np.concatenate( (data_reta,data_eratio,data_f1,data_f3,data_weta2, data_wstot), axis=1)

  # split for this sort
  x_train = [ data_rings[splits[sort][0]], data_shower[splits[sort][0]] , data_track[splits[sort][0]]]
  x_val   = [ data_rings[splits[sort][1]], data_shower[splits[sort][1]] , data_track[splits[sort][1]]]
  y_train = target [ splits[sort][0] ]
  y_val   = target [ splits[sort][1] ]

  return x_train, x_val, y_train, y_val, splits, []




#
# Get configuration file
#
def getJobConfigId( path ):
  return dict(load(path))['id']





#
# Model generator
#
class Model( model_generator_base ):

  def __init__( self, rings_path, shower_path, track_path ):

    model_generator_base.__init__(self)
    import tensorflow as tf
    from tensorflow.keras import layers


    # ringer shape
    input_rings = layers.Input(shape=(100,), name='Input_rings')
    conv_rings   = layers.Reshape((100,1))(input_rings)
    conv_rings   = layers.Conv1D( 4, kernel_size=2, name='conv1d_rings_1', activation='relu')(conv_rings)
    conv_rings   = layers.Conv1D( 8, kernel_size=2, name='conv1d_rings_2', activation='relu')(conv_rings)
    conv_output  = layers.Flatten()(conv_rings)


    # shower shapes
    input_shower_shapes = layers.Input(shape=(6,), name='Input_shower_shapes')
    dense_shower_shapes = layers.Dense(5, activation='relu', name='dense_shower_layer')(input_shower_shapes)

    # track variables
    input_track_variables = layers.Input(shape=(3,), name='Input_track_variables')
    dense_track_variables = layers.Dense(5, activation='relu', name='dense_track_variables_layer')(input_track_variables)


    # decision layer
    input_concat = layers.Concatenate(axis=1)([conv_output, dense_shower_shapes, dense_track_variables])
    dense = layers.Dense(16, activation='relu', name='dense_layer')(input_concat)
    #dense = layers.Dense(1,activation='linear', name='output_for_inference', kernel_regularizer='l2', bias_regularizer='l2')(dense)
    dense = layers.Dense(1,activation='linear', name='output_for_inference')(dense)
    output = layers.Activation('sigmoid', name='output_for_training')(dense)

    # Build the model
    self.__model = tf.keras.Model([input_rings, input_shower_shapes, input_track_variables], output, name = "model")

    self.__tuned_rings_models = self.load_models(rings_path)
    self.__tuned_shower_models = self.load_models(shower_path)
    self.__tuned_track_models = self.load_models(track_path)


    # Follow the strategy proposed by werner were we keep these weights free to do the fine tunings
    # since most part of these variables have an stronge relationship (correlation).
    self.__trainable=True



  #
  # Make the fusion
  #
  def __call__( self, sort ):

    from tensorflow.keras.models import clone_model
    # create a new model
    model = clone_model( self.__model )
    MSG_INFO(self, "Target model:" )
    model.summary()

    rings_model = self.get_best_model( self.__tuned_rings_models, sort , 0) # five neurons in the hidden layer
    MSG_INFO( self, "Rings model (right):")
    rings_model.summary()

    shower_model = self.get_best_model( self.__tuned_shower_models, sort , 0) # five neurons in the hidden layer
    MSG_INFO( self, "Shower model (middle):")
    shower_model.summary()

    track_model = self.get_best_model( self.__tuned_track_models, sort , 0) # five neurons in the hidden layer
    MSG_INFO( self, "track model (left):")
    track_model.summary()

    # copy rings
    self.transfer_weights( rings_model    , 'conv1d_layer_1'    , model, 'conv1d_rings_1'    , trainable=self.__trainable)
    self.transfer_weights( rings_model    , 'conv1d_layer_2'    , model, 'conv1d_rings_2'    , trainable=self.__trainable)
    # copy shower
    self.transfer_weights( shower_model    , 'dense_layer'    , model, 'dense_shower_layer'    , trainable=self.__trainable)
    # copy track
    self.transfer_weights( track_model  , 'dense_layer'    , model, 'dense_track_variables_layer' , trainable=self.__trainable)
    return model



try:



  job_id = getJobConfigId( args.configFile )

  outputFile = args.volume+'/tunedDiscr.jobID_%s'%str(job_id).zfill(4) if args.volume else 'test.jobId_%s'%str(job_id).zfill(4)

  targets = [
                ('tight_cutbased' , 'T0HLTElectronT2CaloTight'        ),
                ('medium_cutbased', 'T0HLTElectronT2CaloMedium'       ),
                ('loose_cutbased' , 'T0HLTElectronT2CaloLoose'        ),
                ('vloose_cutbased', 'T0HLTElectronT2CaloVLoose'       ),
                ]


  from saphyra.callbacks import sp
  from saphyra import PatternGenerator
  from sklearn.model_selection import StratifiedKFold
  from saphyra.applications import BinaryClassificationJob
  from saphyra.decorators import Summary, Reference
  from saphyra.utils.plot_generator import plot_training_curves

  # create decorators
  decorators = [Summary(), Reference(args.refFile, targets)]


  model = None
  # Select the correct pattern
  if args.type == 'track':
    getPatterns=getPatterns_track
  elif args.type == 'fusion':
    getPatterns=getPatterns_fusion
    model = Model( args.path_to_rings, args.path_to_shower , args.path_to_track )
  else:
    getPatterns=getPatterns_track


  #
  # Create the binary job
  #
  job = BinaryClassificationJob(  PatternGenerator( args.dataFile, getPatterns ),
                                  StratifiedKFold(n_splits=10, random_state=512, shuffle=True),
                                  job               = args.configFile,
                                  loss              = 'binary_crossentropy',
                                  metrics           = ['accuracy'],
                                  callbacks         = [sp(patience=25, verbose=True, save_the_best=True)],
                                  epochs            = 5000,
                                  #plots             = [plot_training_curves],
                                  class_weight      = False,
                                  model_generator   = model, # Force to work with model generator (Hack)
                                  models            = [None] if args.type=='fusion' else None, # Force to work with model generator (Hack)
                                  outputFile        = outputFile )

  job.decorators += decorators


  #
  # Run it!
  #
  job.run()


  # necessary to work on orchestra
  from saphyra import lock_as_completed_job
  lock_as_completed_job(args.volume if args.volume else '.')
  sys.exit(0)



except Exception as e:
  print(e)
  # necessary to work on orchestra
  from saphyra import lock_as_failed_job
  lock_as_failed_job(args.volume if args.volume else '.')
  sys.exit(1)



