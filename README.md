This repo represented a slimmed-down version of code used for quick and easily modifiable training with discretized neural networks, first developed for A Contextual Discretization framework for compressing Recurrent Neural Networks (https://openreview.net/forum?id=SJGIC1BFe) and furthered for advanced work with Binary Networks in adversarial and deep settings.

This is not meant to be cutting-edge, or to represent the full research code. 
Instead, this is meant to show the easy of use of the QuantManager-based discretization on MNIST, which allows for much quicker deployment and modification of binary networks of all sorts than what was previously offered.

If you have any questions, or would be interested in seeing the full research code, feel free to reach out to me.

### Usage ###

The syntax for use is very straight forward.
In the `code` directory, `BRNN.py` is the primary file.

Run,
	`python BRNN.py <type> <backprop_type> <topology> <batch_number>`

Where `<type>` is one of:
	`--full`			: which results in a full precision network.
	`--BinaryConnect`	: which results in a network with binary weights.
	`--BinaryNet`		: which results in a network with binary weights and activations.

And `<backprop_type>` is one of:
	`--Identity`		: which results in the gradient of tf.sign being set to the identity. This is the default.
	`--STE`				: which results in the gradient of tf.sign being set to the standard Straight-Through-Estimator
	`--Chernoff`		: which results in the gradient of tf.sign being set to my experimental Binomial Estimator (use this for deep networks)

And `topology` is one of:
	`--simple`			: resulting in a shallow network. This is the default.
	`--deep`			: resulting in a deeper network.

And `<batch_number>` should be any integer, which is the number of epochs you would like to run. 5 the is default.
`<type>` has no default and must be specified.

If, at any point after the first command-line argument, you give the flag `--adversarial`, instead of the standard network you will train a small network and create adversarial samples for it.
I would highly encourage you to examine `BRNN.py` and `models.py` to see how simple the code is, as well as see `QuantManager` to understand how the quantification works.

### Requirements ###

Tensorflow and Keras, along with associated standard libraries.
(If for some reason you hate Keras, only a few lines need to be removed to remove the dependency: mostly calls to Dense)
