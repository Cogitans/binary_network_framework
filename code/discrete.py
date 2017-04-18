import numpy as np 
import tensorflow as tf
import pickle, math
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

def train(sess, graph, training_generator, validation_generator, backprop, loss, acc_value, clip_ops, disc_ops, initialization_ops):
	losses, accuracies = [], []
	with sess.as_default(), graph.as_default():
		tf.global_variables_initializer().run()
		sess.run(initialization_ops)
		for batch_iter in range(NUM_BATCH + 1):
			batch = next(training_generator)
			feed_dict = {"x:0": batch[0], "y:0": batch[1], "learning_rate:0": 1e-1 if batch_iter < trigger else 1e-3, "training_phase:0": True}
			sess.run(disc_ops)
			acc_t, _ = sess.run([acc_value, backprop], feed_dict = feed_dict)
			sess.run(clip_ops, feed_dict)
			if batch_iter % HOW_OFTEN == 0 and batch_iter > 0:
				test_batch = next(validation_generator)
				acc_v, loss_v = sess.run([acc_value, loss], {"x:0": test_batch[0], "y:0": test_batch[1], "training_phase:0": False})
				losses.append(loss_v)
				accuracies.append(acc_v)
				print("On iteration {0}, \n\taccuracy = {1}\n\ttraining accuracy = {2}".format(batch_iter, acc_v, acc_t))
		sess.run(disc_ops)
		weights = [(w.name, sess.run(w)) for w in tf.trainable_variables()]
	return weights, losses, accuracies

def fgsm_generation(sess, graph, x_s, adv_y_s, loss, weights, eps = 1e-1):
	x = graph.get_tensor_by_name("x:0")
	update = tf.gradients(loss, x)
	with sess.as_default(), graph.as_default():
		init_op = tf.global_variables_initializer()
		init_op.run()
		sess.run([tf.assign(graph.get_tensor_by_name(w[0]), tf.constant(w[1])) for w in weights])
		feed_dict = {"x:0": x_s, "y:0": adv_y_s, "training_phase:0": False}
		dx = sess.run(update, feed_dict = feed_dict)
	return x_s - eps * np.sign(dx[0])

def predict(x, weights, sess, graph):
	output = graph.get_tensor_by_name("output:0")
	preds = tf.argmax(output, axis=1)
	with sess.as_default(), graph.as_default():
		init_op = tf.global_variables_initializer()
		init_op.run()
		sess.run([tf.assign(graph.get_tensor_by_name(w[0]), tf.constant(w[1])) for w in weights])

		feed_dict = {"x:0": x, "training_phase:0": False}
		preds_v = sess.run(preds, feed_dict = feed_dict)
	return preds_v

def evaluate_adversarial(adv, t_ys, f_ys, weights, sess, graph):
	altered_predictions = predict(adv, weights, sess, graph)
	got_a = np.equal(t_ys, altered_predictions)
	broke_a = np.not_equal(t_ys, altered_predictions)
	tricked_a = np.equal(f_ys, altered_predictions)
	return got_a, broke_a, tricked_a

def save_results(SAVE_PATH, losses, accuracies, name):
	with open(SAVE_PATH + name, "wb") as f:
		pickle.dump([losses, accuracies], f)

def save_weights(SAVE_PATH, weights, name):
	with open(SAVE_PATH + name + ".weights", "wb") as f:
	 	pickle.dump([(w[0], w[1]) for w in weights], f)

#########################################
#########	   Aggragates     	#########
#########################################

def train_simple(b_type, bt = "FancyIdentity", initialization = he_normal()):
	sess, graph = keras_init()	
	network = simple_network(b_type, graph, binarize_input = False, initialization = initialization, backprop_type = bt)
	weights, losses, accuracies = train(sess, graph, mnist_generator(128), mnist_test_generator(), *network)

def train_deep(b_type, bt = "FancyIdentity", initialization = he_normal(), depth = 5):
	sess, graph = keras_init()	
	network = deep_network(b_type, graph, binarize_input = False, backprop_type = bt, initialization = initialization, num_layers = depth)
	weights, losses, accuracies = train(sess, graph, mnist_generator(128), mnist_test_generator(), *network)

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
	print("Your network was tricked {0}% of the time while operating at {1} accuracy".format(np.average(tricked_a), acc))

#########################################
#########	 Run tests here 	#########
#########################################
if __name__ == '__main__':
	assert sys.argv[1] in ["--full", "--BinaryNet", "--BinaryConnect", "--adversarial"]
	b_type = sys.argv[1].split("--")[1]
	if "--adversarial" in sys.argv:
		test_adversarially(b_type)
	else:
		backprop_type = "Identity"
		if len(sys.argv) > 2:
			assert sys.argv[2] in ["--Identity", "--STE". "--Chernoff"]
			backprop_type = sys.argv[2].split("--")[1]
		model_function = train_simple
		if len(sys.argv) > 3:
			assert sys.argv[3] in ["--small", "--deep"]
			model_function = train_simple if sys.argv[3] == "--small" else train_deep
		NUM_EPOCH = 10
		if len(sys.argv) > 4:
			NUM_EPOCH = int(sys.argv[4])
		NUM_BATCH = (50000 / 128) * NUM_EPOCH
		HOW_OFTEN = 1000
		train_feedforward(b_type)