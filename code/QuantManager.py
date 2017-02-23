import tensorflow as tf
from quantifications import *
from utils import *

# This is a custom context manager for neural network discretization.
# Correct usage allows the user to abstract away all discretization ops
# and allow the QuantManager to handle them in the backgroun.
#
# This saved me countless hours, and I believe is the best tool 
# currently available for handling discrete neural networks.
class QuantManager:
	def __init__(self, b_type):
		self.known_variables = []
		self.discrete_variables = []
		self.lookup_table = {}
		self.quant_conds = {}
		self.all = False
		self.clip = [identity, None]
		self.b_type = b_type

	def __enter__(self):
		return self

	def __exit__(self, *args):
		scope_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
		self.register(scope_variables)

	# It might be hacky, but the best way to specify different quantizations 
	# for different weights is by tensorflow-variable name. 
	# 
	# Call this function in order to define rules for quantization (see models.py for details)
	def define_quant(self, str_cond = None, quant_fn = None, val = None, initialization = binary_choice):
		if self.b_type in ["BinaryNet", "GumbleBinaryNet", "BinaryConnect"]:
			if str_cond:
				assert str_cond not in self.quant_conds
				assert not self.all
				self.quant_conds[quant_fn(val)] = [str_cond, val, initialization(val), []] 
			else:
				assert len(self.quant_conds) == 0
				self.all = True 
				self.quant_conds = {"all": [quant_fn(val), val, initialization(val)]}

	# Call this to automatically convert network gradients into updating the
	# full precision weights instead of the discrete ones.
	def handle_gradients(self, old_grads_and_vars):
		new_grads_and_vars = []
		for grad, var in old_grads_and_vars:
			if self.clip[1]:
				new_grad = self.clip[0](grad, -self.clip[1], self.clip[1])
			else:
				new_grad = grad
			if var in self.lookup_table:
				new_grads_and_vars.append((new_grad, self.lookup_table[var]))
			else:
				new_grads_and_vars.append((new_grad, var))
		return new_grads_and_vars

	# Used to determine gradient clipping type and amount in the 
	# function above.
	def register_clip(self, fn, val):
		self.clip = [fn, val]

	# This method returns a list of tensorflow operations to be run once,
	# at the beginning of training, in order to properly set variables.
	def initialization_ops(self):
		ops = []
		if self.all:
			for var in self.lookup_table:
				ops.append(tf.assign(var, self.quant_conds['all'][2](var.get_shape().as_list())))
		else:
			for quant_cond in self.quant_conds:
				for var in self.quant_conds[quant_cond][3]:
					cond_list = self.quant_conds[quant_cond]
					ops.append(tf.assign(var, cond_list[2](var.get_shape().as_list())))
		return ops

	# This method returns a list of tensorflow operations to be run after
	# every backpropagation in order to update the discrete weights base
	# on their real-valued counterparts.
	def rediscretization_ops(self):
		ops = []
		if self.all:
			for var in self.lookup_table:
				ops.append(tf.assign(var, self.quant_conds['all'][0](self.lookup_table[var])))
		else:
			for quant_cond in self.quant_conds:
				for var in self.quant_conds[quant_cond][3]:
					ops.append(tf.assign(var, quant_cond(self.lookup_table[var])))
		return ops

	# This returns TF operations to be run after eveyr backprop, which
	# keep the real valued weights similar to their discrete counterparts.
	def clip_ops(self):
		ops = []
		for quant_cond in self.quant_conds:
				for var in self.quant_conds[quant_cond][3]:
					ops.append(tf.assign(var, tf.clip_by_value(var, -self.quant_conds[quant_cond][1], self.quant_conds[quant_cond][1])))
		return ops

	# Helper method called automatically by the context manager
	# in order to handle all variables defined in the network.
	def register(self, tf_vars):
		not_q = []
		for var in tf_vars:
			name = var.name.split("/")[-1].split(":")[0].split("_")
			quant_type = None
			if not self.all:
				for cond in self.quant_conds:
					if self.quant_conds[cond][0] in name:
						quant_type = self.quant_conds[cond]
						self.quant_conds[cond][3].append(var)
			else:
				quant_type = self.quant_conds["all"][0]
			if quant_type is not None:
				new_var = tf.Variable(var.initialized_value(), dtype=tf.float32, trainable=False)
				self.known_variables.append(var) 
				self.discrete_variables.append(var)
				self.lookup_table[var] = new_var
			else:
				not_q.append(str(var))

		if len(not_q):
			print("These variables will not be quantized:")
		for v in not_q:
			print("\t" + str(v))