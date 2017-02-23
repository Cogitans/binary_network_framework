import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np 
from keras import backend as K

# These are three variants of discretization functions
# Not used in the slimmed down implementation, but feel
# free to mess around with them if you want.
#######################################################
def deterministic_ternary(val):
	def to_ret(x):	
		return tf.select(tf.less(tf.abs(x), tf.constant(val/2, tf.float32)), tf.zeros_like(x), tf.constant(val, tf.float32)*tf.sign(x))
	return to_ret

def stochastic_ternary(val):
	def to_ret(x):
		x_1 = tf.constant(val, tf.float32) * tf.sign(x)
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.select(tf.less(rand, s), x_1, tf.zeros_like(x_1))
	return to_ret

def deterministic_binary(val):
	def to_ret(x):
		return tf.constant(val, tf.float32) * tf.sign(tf.select(tf.equal(x, 0.), tf.ones_like(x), x))
	return to_ret

def stochastic_binary(val):
	def to_ret(x):
		correct = tf.sign(tf.select(tf.equal(x, 0.), tf.ones_like(x), x))
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.constant(val, tf.float32) * tf.select(tf.less(rand, s), correct, -1*correct)
	return to_ret

def identity(x):
	def f(x):
		return x
	return f
#######################################################


# This operation is heavily based on http://stackoverflow.com/questions/39048984/tensorflow-how-to-write-op-with-gradient-in-python
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Backwards propagation through the Straight-Through Estimator
# is treated like backwards propagation through a hard-tanh.
def IdentityGrad(op, grad):
	X = op.inputs[0]
	condition = tf.logical_and(tf.less_equal(X, tf.ones_like(X)), tf.greater_equal(X, -tf.ones_like(X)))
	return tf.where(condition, grad, tf.zeros_like(grad))

# Sigmoidal (0-1) binarization, not used is this code, but feel free to try.
def SigBinarization(x):
	return np.select([x > 0, x <= 0], [np.ones_like(x), np.zeros_like(x)])

# TanH (-1-1) binarization.
def TanHBinarization(x):
	return np.select([x > 0, x <= 0], [np.ones_like(x), -np.ones_like(x)])

# Rounding (0-1) binarization, splitting on 0.5, used for binarizing the input.
def RoundBinarization(x):
	return np.select([x <= .5, x > .5], [np.zeros_like(x), np.ones_like(x)])

# The tensorflow-fu of a Straight-Through Estimator with different behavior
# during the forwards and backwards passes.
def StraightThroughEstimator(x, style = 'tanh', name=None):
	with tf.name_scope(name, "STE", [x]) as name:
		if style == "sigmoid":
			F = SigBinarization 
		elif style == "tanh":
			F = TanHBinarization
		elif style == "round":
			F = RoundBinarization
		else:
			error("Invalid type of binarization.")
		z = py_func(F, [x], [tf.float32], name=name, grad=IdentityGrad)
		x_shape = x.get_shape().as_list()
		z[0].set_shape(x_shape)
		return z[0]
