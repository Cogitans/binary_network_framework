import numpy as np 
import tensorflow as tf
import pickle, math, sys
from quantifications import *
from utils import *
from keras_extensions import *
from models import *
from QuantManager import QuantManager
from keras import backend as K
from keras.metrics import categorical_accuracy as accuracy
from tensorflow.examples.tutorials.mnist import input_data

#####
ROOT = "../results/"
SAVE_PATH = mkdir(ROOT + "slimmed_down_test/") # This is where the weights will be saved
how_often = 25 # How often to report progress.
#####

# Train a neural network. There are many arguments, but the last 6 are all given in the same sequence as they are
# returned from the network creation method (see `train_feedforward` below to see how this is easily handled).
#
# The first four arguments are:
#	sess 				: a tensorflow session
#	graph				: a tensorflow graph which the model was defined in
#	training_generator	:	a generator producing (x, y) pairs for training
#	validation_generator	:	a generator producing (x, y) pairs for validation
def train(sess, graph, training_generator, validation_generator, backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops,):
	losses, accuracies = [], []
	with sess.as_default(), graph.as_default():
		# Set up the network
		init_op = tf.global_variables_initializer()
		init_op.run()

		# Run our initialization operaitons.
		# It's easier to do this than customize the tensorflow ops
		sess.run(initialization_ops)

		# For each batch:
		for batch_iter in range(num_batch):

			batch = training_generator.next() # get new data
			feed_dict = {"x:0": batch[0], "y:0": batch[1]}
			
			# Discretize all weights
			sess.run(disc_ops)
			# Run backprop
			sess.run(backprop, feed_dict = feed_dict)
			# Re-adjust the real-valued weights
			sess.run(clip_ops)

			# Every once in a while test on the validation set.
			if batch_iter % how_often == 0:
				test_batch = validation_generator.next()
				acc_v, loss_v = sess.run([acc_value, loss], {"x:0": test_batch[0], "y:0": test_batch[1]})
				losses.append(loss_v)
				accuracies.append(acc_v)
				if batch_iter % (10 * how_often) == 0:
					print("On iteration {0}, \n\taccuracy = {1}".format(batch_iter, acc_v))
		# Discretize one more time and save the weights
		sess.run(disc_ops)
		weights = [(w.name, sess.run(w)) for w in tf.trainable_variables()]
	return weights, losses, accuracies

# Not used in the example, but feel tree to experiment with.
# Infers class labels for an input matrix x, determined by some
# sess + graph combination (useful if you have multiple variants
# of the same network with different levels of precision).
def predict(x, weights, sess, graph):
	output = graph.get_tensor_by_name("output:0")
	preds = tf.argmax(output, axis=1)

	with sess.as_default(), graph.as_default():
		init_op = tf.global_variables_initializer()
		init_op.run()
		sess.run([tf.assign(w[0], tf.constant(w[1])) for w in weights])

		feed_dict = {"x:0": x}
		preds_v = sess.run(preds, feed_dict = feed_dict)
	return preds_v

# Helper method for saving metrics
def save_results(losses, accuracies, name):
	with open(SAVE_PATH + name, "wb") as f:
		pickle.dump([losses, accuracies], f)

# Helper method for saving weights
def save_weights(weights, name):
	with open(SAVE_PATH + name + ".weights", "wb") as f:
	 	pickle.dump([weights], f)

#########################################
#########	   Aggragates     	#########
#########################################

# Train a feedforward network
def train_feedforward(b_type):
	binarize = True if b_type != "full" else False

	# Intialize keras and tensorflow
	sess, graph = keras_init()

	# Get our training and validation generators, 
	# along with their shapes.
	generator = mnist_generator(128)
	validation_generator = mnist_test_generator()
	_, n_input = generator.next()[0].shape
	_, n_out = generator.next()[1].shape

	# Create the network
	network = feedforward_network(b_type, graph, n_input, n_out, binarize)

	# Train the network
	weights, losses, accuracies = train(sess, graph, generator, validation_generator, *network)

	# Save the learned information
	save_weights(weights, b_type)
	save_results(losses, accuracies, b_type)

#########################################
#########	 Run tests here   	#########
#########################################
if __name__ == '__main__':
	assert sys.argv[1] in ["--full", "--BinaryNet", "--BinaryConnect"]
	b_type = sys.argv[1].split("--")[1]
	num_batch = 1000
	if len(sys.argv) > 2:
		num_batch = int(sys.argv[2])
	train_feedforward(b_type)