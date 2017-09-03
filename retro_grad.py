# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""RetroGrad for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.training import gradient_descent

import tensorflow as tf

class RetroGradOptimizer(gradient_descent.GradientDescentOptimizer):


  def __init__(self, initial_learning_rate, use_locking=False, name="RetroGrad"):
    """Construct a new Adagrad optimizer.

    Args:
      initial_learning_rate: A `Tensor` or a floating point value.  The initial learning rate.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "RetroGrad".

    """
    super(RetroGradOptimizer, self).__init__(initial_learning_rate, use_locking, name)
    self._learning_rate = initial_learning_rate
    # Created in Initialize.
    self._learning_rate_tensor = None
    self.eps = 1e-6

  def _create_slots(self, var_list):
    self.var_list = var_list
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name="learning_rate")
    for v in var_list:
      with ops.colocate_with(v):
        dtype = v.dtype.base_dtype

        init_neg = init_ops.constant_initializer(-1.0, dtype=dtype)
        init1 = init_ops.constant_initializer(1.0, dtype=dtype)

      self._get_or_make_slot_with_initializer(v, init_neg, v.get_shape(), dtype, "prev_delta", self._name)
      self._get_or_make_slot_with_initializer(v, init1, v.get_shape(), dtype, "curr_delta", self._name)


      self._get_or_make_slot_with_initializer(v, init1, v.get_shape(), dtype, "per_var_lr", self._name)


  def _prepare(self):
    self.step_counter = tf.Variable(0)



  def _apply_dense(self, grad, var):

    prev_delta = self.get_slot(var, "prev_delta")
    curr_delta = self.get_slot(var, "curr_delta")
    per_var_lr = self.get_slot(var, "per_var_lr")

    #first make the step
    apply_op = super(RetroGradOptimizer, self)._apply_dense(grad*per_var_lr, var)
    with tf.control_dependencies([apply_op]):

      #Now shift the curr delta to prev
      with tf.control_dependencies([tf.assign(prev_delta, curr_delta)]):

          #Now update the curr delta with the newest step
          with tf.control_dependencies([tf.assign(curr_delta, grad * per_var_lr)]):

              with tf.control_dependencies([tf.assign(self.step_counter, self.step_counter + 1)]):

                  return tf.assign(per_var_lr, per_var_lr * tf.abs(1 + curr_delta / prev_delta))
                  #Now use the formula to update lr.
                  # return tf.cond(tf.less(self.step_counter, 3), lambda: tf.identity(per_var_lr),
                  #                lambda: tf.assign(per_var_lr, per_var_lr * tf.abs(1 + curr_delta / prev_delta)))






  def _resource_apply_dense(self, grad, var):
      assert (False)
      return super(RetroGradOptimizer, self)._resource_apply_dense(grad, var)


  def _apply_sparse(self, grad, var):
      assert (False)
      return super(RetroGradOptimizer, self)._apply_sparse(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
      assert (False)
      return super(RetroGradOptimizer, self)._resource_apply_sparse(grad, var, indices)

