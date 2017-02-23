import numpy as np
import tensorflow as tf
from keras import backend as K
import time, pickle, os

# Used for registering a graph with Keras and
# creating a new session. Redundant in this
# slimmed-down example, but generally needed.
def keras_init():
    g = tf.Graph()
    sess = tf.Session(graph = g)
    K.set_session(sess)
    K.manual_variable_initialization(True)
    return sess, g

# Helper functions for visualization if you want them.
#################################
def printWeights():
    w_list = [(w.eval(), w.name) for w in tf.trainable_variables()]
    for w in w_list:
        print("Variable " + w[1] + " contains the values: " + str(set(w[0].ravel())))

def print_weight_set(sess, tensor, fd):
    print("Tensor " + tensor.name + " has weights in :" + str(set(sess.run(tensor, fd).ravel())))
#################################

# Makes sure a directory exists before you save anythign in it.
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

# These are functions for determining how to initialize
# the variables in a discrete network. Only binary_choice is used here, 
# but feel free to experiment with the others.
#################################
def ternary_choice(val):
    def init(shape, scale=None, name=None):
        mat = np.random.choice([0, -val, val], shape)
        return tf.Variable(mat, dtype=tf.float32, name=name)
    return init

def binary_choice(val):
    def init(shape, scale=None, name=None):
        mat = np.random.choice([-val, val], shape)
        return tf.Variable(mat, dtype=tf.float32, name=name)
    return init

def scale_identity(val):
    def init(shape, scale=None, name=None):
        mat = np.eye(*shape) * val
        return tf.Variable(mat, dtype=tf.float32, name=name)
    return init
#################################

# Helper functions for MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def mnist_generator(batch_size):
    while True:
        yield mnist.train.next_batch(batch_size)

def mnist_test_generator():
    while True:
        yield mnist.test.next_batch(5000)