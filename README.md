This repo represented a slimmed-down version of code used for quick and easily modifiable training with discretized neural networks, developed for A Contextual Discretization framework for compressing Recurrent Neural Networks (https://openreview.net/forum?id=SJGIC1BFe).

This is not meant to be cutting-edge, or to represent the full research code. 
Instead, this is meant to show the easy of use of the QuantManager-based discretization on MNIST, which allows for much quicker deployment and modification of binary networks of all sorts than what was previously offered.

If you have any questions, or would be interested in seeing the full research code, feel free to reach out to me.

### Usage ###

The syntax for use is very straight forward.
In the `code` directory, `BRNN.py` is the primary file.

Run,
	`python BRNN.py <type> <batch_number>`

Where `<type>` is one of:
	`--full`		: which results in a full precision network.
	`--BinaryConnect`	: which results in a network with binary weights.
	`--BinaryNet`		: which results in a network with binary weights and activations.

And `<batch_number>` should be any integer, which is the number of batches you would like to run.

I would highly encourage you to examine `BRNN.py` and `models.py` to see how simple the code is.

### Requirements ###

Tensorflow and Keras, along with associated standard libraries.
