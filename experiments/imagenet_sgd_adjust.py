# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from resnet import resnet_model
from resnet import vgg_preprocessing
from lr_auto_adjust_sgd_optimizer import SgdAdjustOptimizer

import tf_utils

class Config:
    def __init__(self):
        pass

FLAGS = Config()


_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}


_INITIAL_LEARNING_RATE = None
image_preprocessing_fn = None
network = None
batches_per_epoch = None




def filenames(is_training):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(FLAGS.data_dir, 'train-%05d-of-01024' % i)
        for i in range(0, 1024)]
  else:
    return [
        os.path.join(FLAGS.data_dir, 'validation-%05d-of-00128' % i)
        for i in range(0, 128)]


def dataset_parser(value, is_training):
  """Parse an Imagenet record from value."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = image_preprocessing_fn(
      image=image,
      output_height=network.default_image_size,
      output_width=network.default_image_size,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training):
  """Input function which provides a single batch for train or eval."""
  dataset = tf.contrib.data.Dataset.from_tensor_slices(tf.constant(filenames(is_training)))
  if is_training:
    dataset = dataset.shuffle(buffer_size=1024)
  dataset = dataset.flat_map(tf.contrib.data.TFRecordDataset)

  if is_training:
    dataset = dataset.repeat()

  dataset = dataset.map(lambda value: dataset_parser(value, is_training),
                        num_threads=FLAGS.map_threads,
                        output_buffer_size=FLAGS.batch_size)

  if is_training:
    buffer_size = 1250 + 2 * FLAGS.batch_size
    dataset = dataset.shuffle(buffer_size=buffer_size)

  iterator = dataset.batch(FLAGS.batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


def resnet_model_fn(features, labels, mode):
  """Our model_fn for ResNet to be used with our Estimator."""
  tf.summary.image('images', features, max_outputs=6)

  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We perform weight decay on all trainable
  # variables, which includes batch norm beta and gamma variables.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 120, and 150 epochs.
    if FLAGS.disable_lr_change:
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [30, 60, 120, 150]]
        values = [
            _INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)


    else:
        learning_rate = _INITIAL_LEARNING_RATE

    var_list = (
        tf.trainable_variables() +
        tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

    optimizer = SgdAdjustOptimizer(var_list,
                       base_optimizer=FLAGS.base_optimizer,
                       learning_rate=learning_rate,
                       use_locking=False,
                       name='optimizer',
                       momentum=_MOMENTUM,
                       per_variable=FLAGS.per_variable,
                       iters_per_adjust=FLAGS.iters_per_adjust)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(optimizer.lr, name='learning_rate')
    tf.summary.scalar('learning_rate', optimizer.lr)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    # optimizer = tf.train.MomentumOptimizer(
    #     learning_rate=learning_rate,
    #     momentum=_MOMENTUM)
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    if FLAGS.disable_lr_change == False:
        angle = tf.identity(
            tf.cond(optimizer.u_norm * optimizer.v_norm > 0, lambda: optimizer.lr_update_multiplier, lambda: 0.0),
            name='angle')
        tf.summary.scalar('angle', angle)
    else:
        tf.summary.scalar('angle', 0.0)
  else:
    train_op = None



  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('imagenet_sgd_adjust_optimizer')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il', db_name='imagenet_sgd_adjust_optimizer_db'))

@ex.config
def my_config():
    data_dir = '/home/shai/tensorflow/datasets/imagenet'
    model_dir = tf_utils.allocate_tensorboard_dir() #The directory where the model will be stored
    resnet_size = 50
    train_steps = 6400000
    steps_per_eval = 40000
    #batch_size = 32

    map_threads = 5
    first_cycle_steps = None

    batch_size = 64
    lr = 0.1 * batch_size / 256

    disable_lr_change = False
    if disable_lr_change:
        iters_per_adjust = 999999999
    else:
        iters_per_adjust = int((_NUM_IMAGES['train'] / batch_size)/2)
    per_variable = False
    base_optimizer = 'Momentum'
    seed = 913526365


@ex.automain
@LogFileWriter(ex)
def my_main(data_dir, model_dir, resnet_size, train_steps, steps_per_eval, batch_size, map_threads, first_cycle_steps,
            iters_per_adjust, per_variable, base_optimizer, disable_lr_change, lr, seed):
    global FLAGS
    global _INITIAL_LEARNING_RATE
    global image_preprocessing_fn
    global network
    global batches_per_epoch

    FLAGS.data_dir = data_dir
    FLAGS.model_dir = model_dir
    FLAGS.resnet_size = resnet_size
    FLAGS.train_steps = train_steps
    FLAGS.steps_per_eval = steps_per_eval
    FLAGS.batch_size = batch_size
    FLAGS.map_threads = map_threads
    FLAGS.first_cycle_steps = first_cycle_steps

    FLAGS.iters_per_adjust = iters_per_adjust
    FLAGS.per_variable = per_variable
    FLAGS.base_optimizer = base_optimizer
    FLAGS.disable_lr_change = disable_lr_change


    # Scale the learning rate linearly with the batch size. When the batch size is
    # 256, the learning rate should be 0.1.
    _INITIAL_LEARNING_RATE = lr

    image_preprocessing_fn = vgg_preprocessing.preprocess_image
    network = resnet_model.resnet_v2(
        resnet_size=FLAGS.resnet_size, num_classes=_LABEL_CLASSES)

    batches_per_epoch = _NUM_IMAGES['train'] / FLAGS.batch_size


    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn, model_dir=FLAGS.model_dir)

    for _ in range(FLAGS.train_steps // FLAGS.steps_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy',
            'loss': 'loss',
            'angle': 'angle'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        print('Starting a training cycle.')
        resnet_classifier.train(
            input_fn=lambda: input_fn(True),
            steps=FLAGS.first_cycle_steps or FLAGS.steps_per_eval,
            hooks=[logging_hook])
        FLAGS.first_cycle_steps = None

        print('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(input_fn=lambda: input_fn(False))
        print(eval_results)


