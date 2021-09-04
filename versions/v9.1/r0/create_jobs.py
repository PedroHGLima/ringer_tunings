

from saphyra import *
import tensorflow as tf
from tensorflow.keras import layers

input_rings  = layers.Input(shape=(50,), name='Input_rings')
dense_rings = layers.Dense(5, activation='relu', name='dense_rings_layer')(input_rings)
input_shower_shapes = layers.Input(shape=(6,), name='Input_showers')
dense_shower_shapes = layers.Dense(5, activation='relu', name='dense_shower_layer')(input_shower_shapes)
input_concat = layers.Concatenate(axis=1)([dense_rings, dense_shower_shapes])
dense = layers.Dense(5, activation='relu', name='dense_layer')(input_concat)
dense = layers.Dense(1,activation='linear', name='output_for_inference')(dense)
output = layers.Activation('sigmoid', name='output_for_training')(dense)

# Build the model
model = tf.keras.Model([input_rings, input_shower_shapes], output, name = "model")




create_jobs( models = [model],
        nInits        = 5,
        nInitsPerJob  = 1,
        sortBounds    = 10,
        nSortsPerJob  = 1,
        nModelsPerJob = 1,
        outputFolder  = 'job_config.Zee_v9.1.10sorts.5inits.r0' )


