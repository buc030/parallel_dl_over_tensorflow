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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import ptb_reader
import random
import scipy
from tensorflow.python.client import device_lib

from lr_auto_adjust_sgd import SgdAdjustOptimizer

def data_type():
  return tf.float32

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = ptb_reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self.config = config
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    #self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    if config.normalize_gradients == False:
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    else:
        grads = tf.gradients(self._cost, tvars)

    optimizer = SgdAdjustOptimizer(self._cost, None, tvars,
                                   lr=config.learning_rate,
                                   iters_per_adjust=config.iters_per_adjust,
                                   per_variable=config.per_variable,
                                   iters_to_wait_before_first_collect=config.iters_to_wait_before_first_collect,
                                   base_optimizer=config.base_optimizer,
                                   beta1=config.beta1,
                                   beta2=config.beta2,
                                   rho=config.rho,
                                   lr_update_formula_risky=config.lr_update_formula_risky,
                                   step_size_anealing=config.step_size_anealing,
                                   momentum=config.momentum,
                                   grads_vars=zip(grads, tvars),
                                   normalize_gradients=config.normalize_gradients,
                                   reduce_lr_only=config.reduce_lr_only,
                                   update_every_two_snapshots=config.update_every_two_snapshots)
    self.optimizer = optimizer

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(optimizer.lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):

    def act(x):
        return tf.tanh(x)

    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training, activation=act)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0, activation=act)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    cell = self._get_lstm_cell(config, is_training)
    if is_training and config.keep_prob < 1:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  # max_max_epoch = 26
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = CUDNN


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


def run_epoch(session, model, is_train=False, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    if is_train:
        model.optimizer.run_iter(session, feed_dict)

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, model.config.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)

def get_config(model, rnn_mode, num_gpus, data_path, tensorboard_dir,
            iters_per_adjust, per_variable, iters_to_wait_before_first_collect, base_optimizer,
            beta1, beta2, rho, lr_update_formula_risky, step_size_anealing, momentum, lr, normalize_gradients, max_grad_norm, reduce_lr_only,
               update_every_two_snapshots):

  """Get model config."""
  config = None
  if model == "small":
    config = SmallConfig()
  elif model == "medium":
    config = MediumConfig()
  elif model == "large":
    config = LargeConfig()
  elif model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", model)
  if rnn_mode:
    config.rnn_mode = rnn_mode

  config.learning_rate = lr
  config.reduce_lr_only = reduce_lr_only
  config.max_grad_norm = max_grad_norm
  config.update_every_two_snapshots = update_every_two_snapshots

  assert (num_gpus == 1)
  config.num_gpus = num_gpus
  if num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC

  config.iters_per_adjust, config.per_variable, config.iters_to_wait_before_first_collect, config.base_optimizer, \
    config.beta1, config.beta2, config.rho, config.lr_update_formula_risky, config.step_size_anealing, config.momentum = \
        iters_per_adjust, per_variable, iters_to_wait_before_first_collect, base_optimizer, \
            beta1, beta2, rho, lr_update_formula_risky, step_size_anealing, momentum

  config.normalize_gradients = normalize_gradients
  return config


from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('ptb_sgd_adjust')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il', db_name='ptb_sgd_adjust_db'))

import tf_utils

@ex.config
def my_config():
    model = 'small'
    rnn_mode = CUDNN
    num_gpus = 1

    data_path = '/home/shai/tensorflow/datasets/ptb/data'
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()

    normalize_gradients = True
    iters_per_adjust = 2323
    per_variable=False
    iters_to_wait_before_first_collect=0
    base_optimizer='SGD'
    beta1=0.9
    beta2=0.999
    rho=0.95
    lr_update_formula_risky=True
    step_size_anealing=False
    momentum=0.9
    lr = 1.0
    reduce_lr_only = False
    seed = 913526365

    cut_learning_rates = True
    update_every_two_snapshots = False

    max_grad_norm = 5

    name = 'noname'


@ex.automain
@LogFileWriter(ex)
def my_main(model, rnn_mode, num_gpus, data_path, tensorboard_dir, normalize_gradients, reduce_lr_only,
            iters_per_adjust, per_variable, iters_to_wait_before_first_collect, base_optimizer, update_every_two_snapshots,
            beta1, beta2, rho, lr_update_formula_risky, step_size_anealing, momentum, lr, cut_learning_rates, max_grad_norm, seed):
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  scipy.random.seed(seed)


  if not data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), num_gpus))

  raw_data = ptb_reader.ptb_raw_data(data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config(model, rnn_mode, num_gpus, data_path, tensorboard_dir,
            iters_per_adjust, per_variable, iters_to_wait_before_first_collect, base_optimizer,
            beta1, beta2, rho, lr_update_formula_risky, step_size_anealing, momentum, lr, normalize_gradients, max_grad_norm, reduce_lr_only, update_every_two_snapshots)
  eval_config = get_config(model, rnn_mode, num_gpus, data_path, tensorboard_dir,
            iters_per_adjust, per_variable, iters_to_wait_before_first_collect, base_optimizer,
            beta1, beta2, rho, lr_update_formula_risky, step_size_anealing, momentum, lr, normalize_gradients, max_grad_norm, reduce_lr_only, update_every_two_snapshots)
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    scipy.random.seed(seed)

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      train_perplexity_ph = tf.placeholder(tf.float32)
      tf.summary.scalar("Training Perplexity", train_perplexity_ph, ['per_epoch_summaries'])
      tf.summary.scalar("Learning Rate", m.optimizer.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

      valid_perplexity_ph = tf.placeholder(tf.float32)
      tf.summary.scalar("Validation Perplexity", valid_perplexity_ph, ['per_epoch_summaries'])
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    per_epoch_summaries = tf.summary.merge_all('per_epoch_summaries')
    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    if tf.__version__ < "1.1.0" and num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False

    sv = tf.train.Supervisor(logdir=tensorboard_dir, global_step=m.optimizer.sgd_counter)
    m.optimizer.set_summary_writer(sv._summary_writer)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        if cut_learning_rates:
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            m.optimizer.lrs = [config.learning_rate * lr_decay, config.learning_rate * lr_decay, config.learning_rate * lr_decay]

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.optimizer.lr)))

        #train a single epoch
        train_perplexity = run_epoch(session, m, is_train=True,
                                     verbose=True)


        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        sv._summary_writer.add_summary(session.run(per_epoch_summaries, {valid_perplexity_ph : valid_perplexity, train_perplexity_ph : train_perplexity}), i)

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if tensorboard_dir:
        print("Saving model to %s." % tensorboard_dir)
        sv.saver.save(session, tensorboard_dir, global_step=sv.global_step)
