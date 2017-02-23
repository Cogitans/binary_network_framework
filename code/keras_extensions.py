import numpy as np 
import tensorflow as tf 
from quantifications import *
from keras.layers.core import Dense

# A nice abstraction barrier for BinaryConnect/Net
# This lets you define the same feedforward model
# for both full precision and limited-precision
# architectures, without worrying about if-statements
# and extra layers. 
def DiscreteDense(dim, b_type, activation = None):
    def f(x):
        activation = 'tanh' if b_type != "BinaryNet" else None
        layer = Dense(dim, activation = activation)(x)
        if b_type == "BinaryNet":
            layer = StraightThroughEstimator(layer)
        return layer
    return f
