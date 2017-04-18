import numpy as np 
import tensorflow as tf
import pickle, math, sys
from quantifications import *
from utils import *
from tf_extensions import *
from models import *
from keras.metrics import categorical_accuracy as accuracy
from tensorflow.examples.tutorials.mnist import input_data
#####
ROOT = "../results"
SAVE_PATH = mkdir(ROOT + "slimmed_down_test/")
#####

# Train a neural network. There are many arguments, but the last 6 are all given in the same sequence as they are
# returned from the network creation method (see `train_feedforward` below to see how this is easily handled).
#
# The first four arguments are:
#	sess 				: a tensorflow session
#	graph				: a tensorflow graph which the model was defined in
#	training_generator	:	a generator producing (x, y) pairs for training
#	validation_generator	:	a generator producing (x, y) pairs for validation
def train(sess, graph, training_generator, validation_generator, backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops):
	losses, accuracies = [], []
	with sess.as_default(), graph.as_default():

		# Initialize variables, and run any "re"-initializations needed for quantification.
		tf.global_variables_initializer().run()
		sess.run(initialization_ops)

		# For each batch:
		for batch_iter in range(NUM_BATCH + 1):

			# Get the next batch of data and labels and set up the feed_dict
			batch = next(training_generator)
			feed_dict = {"x:0": batch[0], "y:0": batch[1], "learning_rate:0": 1e-3}

			# Run discretization operations before optimization step.
			sess.run(disc_ops)

			# Do the ADAM update, saving the training accuracy
			acc_t, _ = sess.run([acc_value, backprop], feed_dict = feed_dict)

			# Run the clipping operations, maintaining the full precision variables
			sess.run(clip_ops, feed_dict)

			# Every so often:
			if batch_iter % HOW_OFTEN == 0 and batch_iter > 0:

				# Get the next batch (there's only one batch) of validation data and labels
				test_batch = next(validation_generator)

				# Save the validation loss and accuracy
				acc_v, loss_v = sess.run([acc_value, loss], {"x:0": test_batch[0], "y:0": test_batch[1]})
				losses.append(loss_v)
				accuracies.append(acc_v)
				print("On iteration {0}, \n\taccuracy = {1}\n\ttraining accuracy = {2}".format(batch_iter, acc_v, acc_t))

		# Run discretization one more time.
		sess.run(disc_ops)

		# Save and return the weights, alongside your loss and accuracy history.
		weights = [(w.name, sess.run(w)) for w in tf.trainable_variables()]
	return weights, losses, accuracies

# A helper method for generating adversarial modifications to the inputs given in x_s,
# based on the "false" labels given in adv_y_s. 
def fgsm_generation(sess, graph, x_s, adv_y_s, loss, weights, eps = 1e-1):
	x = graph.get_tensor_by_name("x:0")
	update = tf.gradients(loss, x)
	with sess.as_default(), graph.as_default():
		init_op = tf.global_variables_initializer()
		init_op.run()
		sess.run([tf.assign(graph.get_tensor_by_name(w[0]), tf.constant(w[1])) for w in weights])
		feed_dict = {"x:0": x_s, "y:0": adv_y_s}
		dx = sess.run(update, feed_dict = feed_dict)
	return x_s - eps * np.sign(dx[0])

# Only used for adversarial generation, but feel tree to experiment with.
# Infers class labels for an input matrix x, determined by some
# sess + graph combination (useful if you have multiple variants
# of the same network with different levels of precision).
def predict(x, weights, sess, graph):
	output = graph.get_tensor_by_name("output:0")
	preds = tf.argmax(output, axis=1)
	with sess.as_default(), graph.as_default():
		init_op = tf.global_variables_initializer()
		init_op.run()
		sess.run([tf.assign(graph.get_tensor_by_name(w[0]), tf.constant(w[1])) for w in weights])

		feed_dict = {"x:0": x}
		preds_v = sess.run(preds, feed_dict = feed_dict)
	return preds_v

# A helper function for evaluating the effectiveness of the adversarial samples.
def evaluate_adversarial(adv, t_ys, f_ys, weights, sess, graph):
	altered_predictions = predict(adv, weights, sess, graph)
	got_a = np.equal(t_ys, altered_predictions)
	broke_a = np.not_equal(t_ys, altered_predictions)
	tricked_a = np.equal(f_ys, altered_predictions)
	return got_a, broke_a, tricked_a

# Helper method for saving metrics
def save_results(losses, accuracies, name):
	with open(SAVE_PATH + name, "wb") as f:
		pickle.dump([losses, accuracies], f)

# Helper method for saving weights
def save_weights(weights, name):
	with open(SAVE_PATH + name + ".weights", "wb") as f:
		pickle.dump(weights, f)


#########################################
#########	   Aggragates     	#########
#########################################

# Helper method for training a small feedforward network with specifiable weight initialization and backpropagation.
def train_simple(b_type, bt = "Identity", initialization = he_normal()):
	sess, graph = keras_init()	
	network = simple_network(b_type, graph, binarize_input = False, initialization = initialization, backprop_type = bt)
	weights, losses, accuracies = train(sess, graph, mnist_generator(128), mnist_test_generator(), *network)

# Helper method for training a deeper feedforward network with specifiable weight initialization and backpropagation.
def train_deep(b_type, bt = "Identity", initialization = he_normal(), depth = 15):
	sess, graph = keras_init()	
	network = deep_network(b_type, graph, binarize_input = False, backprop_type = bt, initialization = initialization, num_layers = depth)
	weights, losses, accuracies = train(sess, graph, mnist_generator(128), mnist_test_generator(), *network)

# Helper method for training a small feedforward network and generating adversarial samples on the trained network.
def test_adversarially(b_type, num_iter = 1, eps = 1e-1):
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	x, true_y = mnist.test.next_batch(5000)
	adv_y = adversarial_label_generation(true_y)
	start = np.random.choice(4950)

	sess, graph = keras_init()
	network = simple_network(b_type, graph)
	backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops = network
	weights, losses, accuracies = train(sess, graph, mnist_generator(128), mnist_test_generator(), *network)
	t_ys, f_ys = np.argmax(true_y, axis=1), np.argmax(adv_y, axis=1)
	original_predictions = predict(x, weights, sess, graph)
	acc_v = np.average(np.equal(t_ys, original_predictions))
	all_n = network + [acc_v, t_ys, f_ys, weights, sess, graph]
	adv_examples = x
	for _ in range(num_iter):
	 	adv_examples = fgsm_generation(sess, graph, adv_examples, adv_y, loss, weights, eps)
	got_a, broke_a, tricked_a = evaluate_adversarial(adv_examples, *all_n[-5:])
	print("Your network was tricked {0}% of the time.".format(np.average(tricked_a)))

#########################################
#########	 Run tests here 	#########
#########################################
if __name__ == '__main__':
	assert sys.argv[1] in ["--full", "--BinaryNet", "--BinaryConnect", "--adversarial"]
	b_type = sys.argv[1].split("--")[1]
	if "--adversarial" in sys.argv:
		NUM_BATCH = (50000 / 128) * 5
		HOW_OFTEN = 50
		test_adversarially(b_type)
	else:
		backprop_type = "Identity"
		if len(sys.argv) > 2:
			assert sys.argv[2] in ["--Identity", "--STE", "--Chernoff"]
			backprop_type = sys.argv[2].split("--")[1]
		model_function = train_simple
		if len(sys.argv) > 3:
			assert sys.argv[3] in ["--small", "--deep"]
			model_function = train_simple if sys.argv[3] == "--small" else train_deep
		NUM_EPOCH = 5
		if len(sys.argv) > 4:
			NUM_EPOCH = int(sys.argv[4])
		NUM_BATCH = (50000 / 128) * NUM_EPOCH
		HOW_OFTEN = 50
		model_function(b_type)