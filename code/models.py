import numpy as np 
import tensorflow as tf 
from QuantManager import QuantManager
from keras_extensions import * 
from keras.metrics import categorical_accuracy as accuracy
from quantifications import *
from utils import *

# Creates a feedforward network in the specified graph.
# The arguments are as follows:
# 	b_type			: Either "full", "BinaryNet" or "BinaryConnect", specifying the discretization.
#	graph			: A Tensorflow graph, not really relevant in this version of the code
#	binarize_input	: If b_type 1= "full", whether to binarize the input before the first layer.
def feedforward_network(b_type, graph, n_input, n_out, binarize_input = False):
	with graph.as_default():
		assert b_type in ["full", "BinaryNet", "BinaryConnect"]

		# Define the context manager for discretization
		quant_manager = QuantManager(b_type)
		quant_manager.define_quant("W", quant_fn = deterministic_binary, val = 1.0)
		quant_manager.define_quant("b", quant_fn = deterministic_binary, val = 1.0)

		# Define input/outputs
		x = tf.placeholder(tf.float32, shape=(None, n_input), name="x")
		labels = tf.placeholder(tf.float32, shape=(None, n_out), name="y")

		# Define the network, pay attention to the quant_manager 
		with tf.variable_scope("feedforward"), quant_manager:
			if binarize_input and b_type != "full":
				x = StraightThroughEstimator(x, style = "round")
			h1 = DiscreteDense(400, b_type = b_type)(x)
			h2 = DiscreteDense(300, b_type = b_type)(h1)
			o = Dense(n_out, name="output")(h2)

		# Define training metrics
		output = tf.identity(o, name="output")
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, labels))
		acc_value = accuracy(labels, output)	

		# Define optimizer and gradient update
		# Notice how the quant_manager takes care of all discretization behind the scenes
		optimizer = tf.train.AdamOptimizer(1e-3)
		quant_manager.register_clip(tf.clip_by_value, 10)
		grads_vars = quant_manager.handle_gradients(optimizer.compute_gradients(loss))

		# These three lists of tf operations, generated by the quant_manager,
		# are necessary for correct operation of the graph.
		clip_ops = quant_manager.clip_ops()
		disc_ops = quant_manager.rediscretization_ops()
		initialization_ops = quant_manager.initialization_ops()

		backprop = optimizer.apply_gradients(grads_vars)
	return [backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops]