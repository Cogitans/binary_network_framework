import tensorflow as tf
from quantifications import *
from utils import *

# This is a custom context manager for neural network discretization.
# Correct usage allows the user to abstract away all discretization ops
# and allow the QuantificationManager to handle them in the backgroun.
#
# This saved me countless hours, and I believe is the best tool 
# currently available for handling discrete neural networks in Tensorflow.
class QuantificationManager:

	# A little hacky; its nice to be able to statically access an active manager without passing around a reference.
	# A hitherto unencountered scenario with multiple managers will require something smarter.
	active_manager = None
	acceptable_quantifications = ["BinaryConnect", "BinaryNet"]

	def __init__(self, b_type):
		QuantificationManager.active_manager = self
		self.b_type = b_type
		self.quant_conds = {}
		self.initializations = {}
		self.to_clip = False
		self.quantized_variables_map = {}
		self.tf_scope = None

	# Necessary for managing to work
	def __enter__(self):
		return self

	# When we're finished with a context, we want to exit out of a variable scope
	# (if one was opening alongside this manager), collect all variables defined in 
	# that scope and register them as variables to be quantized.
	def __exit__(self, *args):
		S = tf.get_variable_scope().name
		if self.tf_scope:
			self.tf_scope.__exit__(*args)
			self.tf_scope = None
		scope_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = S)
		self.register(scope_variables)

	# Open up a tensorflow variable scope with the given name and begin monitoring.
	def monitor_tf_variable_scope(self, name):
		self.tf_scope = tf.variable_scope(name)
		self.tf_scope.__enter__()
		return self

	# Take the list of passed in variables and create full-precision copies; tracking pointers
	# and intializations alongside everything needed for training.
	def register(self, tf_vars):
		not_quantized = []
		for var in tf_vars:
			quant_type = None
			for str_cond in self.quant_conds:
				if self.quant_conds[str_cond][1]:
					full_name = var.name.split("/")[-1].split(":")[0]
					is_match = str_cond == full_name
				else:
					is_match = str_cond in var.name
				
				if is_match:
					quant_type = str_cond
					break
			if quant_type is not None and self.b_type in self.acceptable_quantifications:
				if str_cond in self.initializations:
					initialization = self.initializations[str_cond]
				else:
					error("You have attempted to discretize a variable without specifying an inital value.")

				initial_value = initialization(var.get_shape().as_list())
				full_precision_var = tf.Variable(initial_value, dtype=tf.float32, trainable = False, name = var.name.split(":")[0] + "_full_precision")
				self.quantized_variables_map[var] = (full_precision_var, initial_value, self.quant_conds[str_cond][0])
			else:
				not_quantized.append(str(var))

		if len(not_quantized):
			print("These variables will not be quantized:")
			for v in not_quantized:
				print("\t" + str(v))

	# Used before model definition: specifying how variables with "str_cond" in their name should be quantified and initialized
	# If is_full_name is true, the str_cond "name" must be the full name of the variable to match, not just a subset.
	def define_quantification_condition(self, str_cond, quantification_function, is_full_name = False, initialization = None):
		self.quant_conds[str_cond] = (quantification_function, is_full_name)
		if initialization is not None:
			self.define_initialization(str_cond, initialization)

	# Used if you want to define the initialization without a quantification; not currently used.
	def define_initialization(self, str_cond, initialization):
		self.initializations[str_cond] = initialization

	# Defines the numerical amount by which you wish to clip gradients
	def enable_clipping(self, value):
		self.to_clip = value

	# Used to intercept (grad, var) pairs in the optimizer and redirect all gradients to their full precision counterparts.
	# Returns the list which can be continued to be used by the optimizer.
	def handle_gradients(self, old_grads_and_vars, binarize_gradients = False):
		new_grads_and_vars = []
		for i, (grad, var) in enumerate(old_grads_and_vars):
			if binarize_gradients:
				new_grad = deterministic_binary(1.0)(grad)
			elif self.to_clip != False:
				new_grad = tf.clip_by_value(grad, -self.to_clip, self.to_clip)
			else:
				new_grad = grad

			if var in self.quantized_variables_map:
				new_grads_and_vars.append((new_grad, self.quantized_variables_map[var][0]))
			else:
				new_grads_and_vars.append((new_grad, var))
		return new_grads_and_vars

	# Returns the list of ops needed for proper initialization.
	def initialization_ops(self):
		ops = []
		for var in self.quantized_variables_map:
			to_assign = self.quantized_variables_map[var][1]
			ops.append(tf.assign(var, to_assign))
		return ops

	# Returns the list of ops needed to maintain the discrete variables.
	def rediscretization_ops(self):
		ops = []
		for var in self.quantized_variables_map:
			q_cond = self.quantized_variables_map[var][2]
			assign_op = tf.assign(var, q_cond(self.quantized_variables_map[var][0]))
			ops.append(assign_op)
		return ops

	# Returns the list of ops needed to clip the full precision weights.
	def clip_full_precision_ops(self):
		ops = []
		for var in self.quantized_variables_map:
			full_p_var = self.quantized_variables_map[var][0]
			ops.append(tf.assign(full_p_var, tf.clip_by_value(full_p_var, -1.0, 1.0)))
		return ops

	# Not currently used, but gives a list of full precision weights.
	def get_full_precision(self):
		return [self.quantized_variables_map[v][0] for v in self.quantized_variables_map]

	# Not current used, but gives a list of variables this manager is tracking.
	def get_weights(self):
		return [v for v in self.quantized_variables_map]

