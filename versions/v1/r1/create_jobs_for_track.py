
from saphyra import *
import tensorflow as tf
from tensorflow.keras import layers



input  = layers.Input(shape=(3,), name = 'Input')
dense  = layers.Dense(5, activation='relu', name='dense_layer')(input)
dense  = layers.Dense(1,activation='linear', name='output_for_inference')(dense)
output = layers.Activation('sigmoid', name='output_for_training')(dense)
model = tf.keras.Model(input, output, name = "model")


create_jobs( models = [model],
        nInits        = 10,
        nInitsPerJob  = 1,
        sortBounds    = 10,
        nSortsPerJob  = 1,
        nModelsPerJob = 1,
        outputFolder  = 'job_config.Zee_v1_el_trk.10sorts.10inits.r1' )


