import numpy as np 
import tensorflow as tf 
from quantifications import *
from utils import *
from QuantManager import *
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import GRU
from keras.layers import Convolution2D

# A feedforward layer that replaces the tanh activation with a binarization if b_type == "BinaryNet"
def DiscreteDense(dimension, graph, b_type, name, backprop_type = "Identity"):
    def f(x):
        if backprop_type == "Chernoff":
            # The Chernoff backpropgation has to be redefined per layer size
            # This could potentially be a no-op if this size has been encountered.
            b_t = chernoff(x.get_shape().as_list()[-1])
        else:
            b_t = backprop_type

        # Do a standard matrix-mult
        layer = Dense(dimension, name = name)(x)
        if b_type == "BinaryNet":
            # If we're binary, deterministically binarize and 
            # set our backpropagation to be what we want
            with graph.gradient_override_map({"Sign": b_t}):
                layer = deterministic_binary(1.0)(layer)
        else:
            # Otherwise, we are a normal layer.
            layer = Activation("tanh")(layer)
        return layer
    return f
