
from saphyra import *
import tensorflow as tf
from tensorflow.keras import layers



# ringer shape
input  = layers.Input(shape=(100,), name='Input')
dense  = layers.Dense(5, activation='relu', name='dense')(input)
output = layers.Activation('sigmoid', name='output_for_training')(dense)
dummy = tf.keras.Model([input], output, name = "model")



create_jobs( models = [dummy],
        nInits        = 10,
        nInitsPerJob  = 1,
        sortBounds    = 10,
        nSortsPerJob  = 1,
        nModelsPerJob = 1,
        outputFolder  = 'job_config.Zee_v2_el.10sorts.10inits.r1' )


