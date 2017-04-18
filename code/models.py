import numpy as np 
import tensorflow as tf 
from QuantManager import QuantificationManager
from keras.metrics import categorical_accuracy as accuracy
from tf_extensions import * 
from quantifications import *
from utils import *

# A simple, small, feedforward network.
# The arguments are as follows:
# 	b_type			: Either "full", "BinaryNet" or "BinaryConnect", specifying the discretization.
#	graph			: A Tensorflow graph, not really relevant in this version of the code
#	binarize_input	: If b_type 1= "full", whether to binarize the input before the first layer.
# 	backprop_type 	: One of "Identity", "Chernoff", "STE"; describing how to backpropagate through tf.sign
#	initialization 	: A function to initialize your weight matrices with.
def simple_network(b_type, graph, binarize_input = False, backprop_type = "Identity", initialization = he_normal()):
	with graph.as_default():
		assert b_type in ["full", "BinaryNet", "BinaryConnect"]

		# Define the context manager for quantification and set up its conditions.
		qm = QuantificationManager(b_type)
		qm.enable_clipping(10.0)
		qm.define_quantification_condition("W", deterministic_binary(1.0), initialization = initialization)
		qm.define_quantification_condition("b", deterministic_binary(1.0), initialization = binary_choice(1.0))

		# Define variables for use in training
		x = tf.placeholder(tf.float32, shape = (None, 784), name = "x")
		labels = tf.placeholder(tf.float32, shape=(None, 10), name = "y")
		global_step = tf.Variable(0, trainable = False, name = "global_step")
		learning_rate = tf.placeholder(tf.float32, name = "learning_rate")
		actual_rate = tf.train.exponential_decay(learning_rate, global_step, 250, 0.95)

		# Open a tf.Variable_scope which our QuantificationManager will be monitoring
		with qm.monitor_tf_variable_scope("feedforward"):
			# Binarize input if you want
			if binarize_input and b_type != "full":
				with graph.gradient_override_map({"Round": "Identity"}):
					x = tf.round(x)
			h1 = DiscreteDense(256, graph, b_type = b_type, name = "first_hidden", backprop_type = backprop_type)(x)
			o = Dense(10, name="output")(h1)

		# Gather ops useful for forward propagations
		output = tf.identity(o, name="output")
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = labels))
		acc_value = accuracy(labels, output)	

		# Gather ops useful for backward propagations
		optimizer = tf.train.AdamOptimizer(actual_rate)
		grads_vars = qm.handle_gradients(optimizer.compute_gradients(loss))

		# Receive the necessary operations from the QuantificationManager
		clip_ops = qm.clip_full_precision_ops() + [tf.assign(global_step, global_step + tf.constant(1))]
		disc_ops = qm.rediscretization_ops()
		initialization_ops = qm.initialization_ops()

		# Receive the training op
		backprop = optimizer.apply_gradients(grads_vars)
	return [backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops]




# A deeper network which automatically uses the Chernoff backprop.
# The arguments are as follows:
# 	b_type			: Either "full", "BinaryNet" or "BinaryConnect", specifying the discretization.
#	graph			: A Tensorflow graph, not really relevant in this version of the code
#	binarize_input	: If b_type 1= "full", whether to binarize the input before the first layer.
# 	backprop_type 	: One of "Identity", "Chernoff", "STE"; describing how to backpropagate through tf.sign
#	initialization 	: A function to initialize your weight matrices with.
#   num_layers 		: How many hidden layers after the initial layer.
def deep_network(b_type, graph, binarize_input = False, backprop_type = "Chernoff", initialization = he_normal(), num_layers = 10):
	with graph.as_default():
		assert b_type in ["full", "BinaryNet", "BinaryConnect"]

		# Define the context manager for quantification and set up its conditions.
		qm = QuantificationManager(b_type)
		qm.enable_clipping(10.0)
		qm.define_quantification_condition("W", deterministic_binary(1.0), initialization = initialization)
		qm.define_quantification_condition("b", deterministic_binary(1.0), initialization = binary_choice(1.0))

		# Define variables for use in training
		x = tf.placeholder(tf.float32, shape=(None, 784), name = "x")
		labels = tf.placeholder(tf.float32, shape=(None, 10), name = "y")
		global_step = tf.Variable(0, trainable=False, name = "global_step")
		learning_rate = tf.placeholder(tf.float32, name = "learning_rate")
		actual_rate = tf.train.exponential_decay(learning_rate, global_step, 250, 0.95)

		# Open a tf.Variable_scope which our QuantificationManager will be monitoring
		with qm.monitor_tf_variable_scope("feedforward"):
			# Binarize input if you want
			if binarize_input and b_type != "full":
				with graph.gradient_override_map({"Round": "Identity"}):
					x = tf.round(x)
			h1 = DiscreteDense(256, graph, b_type = b_type, name = "first_hidden", backprop_type = backprop_type)(x)
			for _ in range(num_layers):
				h1 = DiscreteDense(128, graph, b_type = b_type, name = "{0}_hidden".format(_ + 2), backprop_type = backprop_type)(h1)
			o = Dense(10, name="output")(h1)

		# Gather ops useful for forward propagations
		output = tf.identity(o, name="output")
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = labels))
		acc_value = accuracy(labels, output)	

		# Gather ops useful for backward propagations
		optimizer = tf.train.AdamOptimizer(actual_rate)
		grads_vars = qm.handle_gradients(optimizer.compute_gradients(loss))

		# Receive the necessary operations from the QuantificationManager
		clip_ops = qm.clip_full_precision_ops() + [tf.assign(global_step, global_step + tf.constant(1))]
		disc_ops = qm.rediscretization_ops()
		initialization_ops = qm.initialization_ops()

		# Receive the training op
		backprop = optimizer.apply_gradients(grads_vars)
	return [backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops]
