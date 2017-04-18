import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np 
from keras import backend as K
from tensorflow.python.framework.function import *
import math

# Variations of the standard discretization, not used in this example.
# Feel free to experiment with them, however.
#######################################################################3
def deterministic_ternary(val):
	def to_ret(x):	
		return tf.select(tf.less(tf.abs(x), tf.constant(val/2, tf.float32)), tf.zeros_like(x), tf.constant(val, tf.float32)*tf.sign(x))
	return to_ret

def stochastic_ternary(val):
	def to_ret(x):
		x_1 = tf.constant(val, tf.float32) * tf.sign(x)
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.where(tf.less(rand, s), x_1, tf.zeros_like(x_1))
	return to_ret

def deterministic_binary(val):
	def to_ret(x):
		return tf.constant(val, tf.float32) * tf.sign(tf.where(tf.equal(x, 0.), tf.ones_like(x), x))
	return to_ret

def stochastic_binary(val):
	def to_ret(x):
		correct = tf.sign(tf.where(tf.equal(x, 0.), tf.ones_like(x), x))
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.constant(val, tf.float32) * tf.where(tf.less(rand, s), correct, -1*correct)
	return to_ret
#######################################################################
# It is worth noting that more complicated discretization schemes can be supported with no extra work.

# An exponential quantification function from Recurrent Neural Networks With Limited Numerical Precision
# (https://arxiv.org/abs/1608.06902)
def exponential_quant(val):
	def f(x):
		sign = tf.sign(x)
		temp = tf.div(tf.log(tf.abs(x)), tf.constant(ln2))
		return tf.pow(2.0, tf.round(temp)) * sign
	return f

# You can also use this framework to support different styles of networks. Use this quantification
# to replace all weight matrices with a low-rank approximation (where ratio is the percent of the full
# rank that you wish to use)
def low_rank_approximation(ratio):
	def f(x):
		s, u, v = tf.svd(x)
		num_s = s.get_shape().as_list()[0]
		mask = tf.constant([1.0 if i < num_s * ratio else 0.0 for i in range(num_s)])
		s *= mask
		return tf.matmul(u, tf.matmul(tf.diag(s), tf.transpose(v)))
	return f
#######################################################################
# The improved (and fast) way of handling variations in the gradient

# This is the standard Straight-Through-Estimator from Hinton, used in BinaryNet.
@tf.RegisterGradient("StraightThroughEstimator")
def limitedidentity(op, grad):
	X = op.inputs[0]
	condition = tf.logical_and(tf.less_equal(X, tf.ones_like(X)), tf.greater_equal(X, -tf.ones_like(X)))
	return [tf.where(condition, grad, tf.zeros_like(grad))]
	
# Here is my proposed Binomial Estimator based on a Chernoff approximation. This will achieve signifigantly higher 
# accuracy with deeper networks.
def chernoff(width):
	try:
		@tf.RegisterGradient("Chernoff{0}".format(width))
		def fancyidentity(op, grad):
			X = op.inputs[0]
			N = float(width)
			# The following 5 lines are ugly, but it's just algbera from the Chernoff approximation
			X = X / 2 + (N / 2)
			l_val = 1 - math.sqrt((4.0 * math.log(4)) / N)
			l_val = l_val * N / 2
			u_val = 1 + math.sqrt((6.0 * math.log(4)) / N)
			u_val = u_val * N / 2
			condition = tf.logical_and(tf.less_equal(X, u_val * tf.ones_like(X)), tf.greater_equal(X, l_val * tf.ones_like(X)))
			return [tf.where(condition, grad, tf.zeros_like(grad))]
		return "Chernoff{0}".format(width)
	except KeyError:
		return "Chernoff{0}".format(width)

#######################################################################
# There are also some fancy alternatives you can plat around with.

# Makes all gradients binary at each step -- doesn't work well.
@tf.RegisterGradient("BinaryGradient")
def BinaryGradient(op, grad):
	X = op.inputs[0]
	condition = tf.logical_and(tf.less_equal(X, tf.ones_like(X)), tf.greater_equal(X, -tf.ones_like(X)))
	return [tf.where(condition, tf.sign(grad), tf.zeros_like(grad))]

# Sacrifices numerical simplicity for a nice gradient to learn under.
@tf.RegisterGradient("LeakySoftsign")
def leakySoftSign(op, grad):
	X = op.inputs[0]
	condition = tf.logical_and(tf.less_equal(X, tf.ones_like(X)), tf.greater_equal(X, -tf.ones_like(X)))
	return [tf.where(condition, grad, tf.constant(0.125)*grad)]
