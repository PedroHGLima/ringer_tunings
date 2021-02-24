

from saphyra import *


def get_model( ):
  modelCol = []
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten
  for n in range(2,10+1):
    model = Sequential()
    model.add(Dense(n, input_shape=(100,), activation='tanh', name='dense_layer'))
    model.add(Dense(1, activation='linear', name='output_for_inference'))
    model.add(Activation('tanh', name='output_for_training'))
    modelCol.append(model)
  return modelCol



create_jobs( models = get_model(),
        nInits        = 10,
        nInitsPerJob  = 1,
        sortBounds    = 10,
        nSortsPerJob  = 1,
        nModelsPerJob = 5,
        outputFolder  = 'job_config.Jpsi_v1.n2to10.10sorts.10inits.r1' )


