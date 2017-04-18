import numpy as np
import tensorflow as tf
from keras import backend as K
import time, pickle, os, math
from tensorflow.examples.tutorials.mnist import input_data

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
# the variables in a discrete network. 
#################################
def ternary_choice(val):
    def init(shape, scale=None, name=None):
        mat = np.random.choice([0, -val, val], shape)
        return tf.constant(mat, dtype=tf.float32, name=name)
    return init

def binary_choice(val):
    def init(shape, scale=None, name=None):
        mat = np.random.choice([-val, val], shape)
        return tf.constant(mat, dtype=tf.float32, name=name)
    return init

def scale_identity(val):
    def init(shape, scale=None, name=None):
        mat = np.eye(*shape) * val
        return tf.constant(mat, dtype=tf.float32, name=name)
    return init

def hadamard():
    def hadmard_init(shape, scale=None, name=None):
        # This is gross - but somewhat necessary. Creating a non power-of-two pseudo-Hadamard
        # matrix takes some work, especially if it is highly non square.
        def make_a_hadamard_mat(shape):
            if shape[-1] == math.pow(2, round(math.log(shape[-1], 2))):
                mat = la.hadamard(shape[-1])
            else:
                next_highest = math.ceil(math.log(shape[-1], 2))
                mat = la.hadamard(math.pow(2, next_highest))
                mat = mat[:, 0:shape[-1]]
            if shape[0] > mat.shape[0]:
                rows = np.random.choice(mat.shape[0], shape[0] - mat.shape[0])
                mat = np.vstack([mat, mat[rows, :]])
            elif shape[0] < mat.shape[0]:
                rows = np.random.choice(mat.shape[0], shape[0], replace = False)
                mat = mat[rows, :]
            return mat
        mat = make_a_hadamard_mat(shape) * math.sqrt(2.0 / shape[0]) * np.abs(np.random.standard_normal(shape))
        return tf.constant(mat, dtype = tf.float32, name = name)
    return hadmard_init

def he_normal():
    def he_init(shape, scale = None, name = None):
        mat = np.random.standard_normal(shape)
        return tf.constant(math.sqrt(2.0 / shape[0]) * mat, dtype=tf.float32, name=name)
    return he_init

#################################
# Helper functions for MNIST dataset and adversarial samples.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def mnist_generator(batch_size):
    while True:
        yield mnist.train.next_batch(batch_size)

def mnist_test_generator():
    while True:
        yield mnist.test.next_batch(5000)

def adversarial_label_generation(y_samples):
    prob_new_y = 1 - y_samples
    new_y = []
    for i in np.arange(y_samples.shape[0]):
        val = np.random.choice(y_samples.shape[1], p = prob_new_y[i, :]/9)
        arr = np.zeros((10,))
        arr[val] = 1
        new_y.append(arr)
    adv_labels = np.array(new_y)
    return adv_labels