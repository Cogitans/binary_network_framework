import numpy as np 
import tensorflow as tf 
from quantifications import *
from utils import *
from QuantManager import *
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import GRU
from keras.layers import Convolution2D

def DiscreteDense(dimension, graph, b_type, name, backprop_type = "FancyIdentity"):
    def f(x):
        if backprop_type == "Chernoff":
            backprop_type = chernoff(x.get_shape().as_list()[-1])
        layer = Dense(dimension, name = name)(x)
        if b_type == "BinaryNet":
            with graph.gradient_override_map({"Sign": backprop_type}):
                layer = deterministic_binary(1.0)(layer)
        else:
            layer = Activation(activation)(layer)
        return layer
    return f
