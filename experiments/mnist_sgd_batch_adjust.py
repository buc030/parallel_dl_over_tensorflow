# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from sgd_batch_adjust import SgdBatchAdjustOptimizer
from batch_provider import MnistBatchProvider

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'accuracy', 'predictions'])

def cnn_model_fn(features, labels, enable_dropout, weighted_batch):
  """Model function for CNN."""
  # Input Layer
  labels = tf.reshape(labels, [-1])
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=enable_dropout)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)


  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

  weights = 1.0
  if weighted_batch:
      weights = tf.concat( [tf.ones(100.0)*(1/200.0), tf.ones([tf.shape(input_layer)[0] - 100])*(tf.ones(1) / tf.cast((2 * (tf.shape(input_layer)[0] - 100)), tf.float32))] , axis=0)* \
                tf.cast(tf.shape(input_layer)[0], tf.float32)

  #This avarage the loss.
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits, weights=weights)

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return Model(loss=loss, accuracy=accuracy, predictions=logits)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('minst_sgd_batch_adjust')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='mnist_sgd_batch_adjust_db'))


@ex.config
def my_config():

    lr = 0.1
    batch_size = 100
    n_epochs = 100

    iters_to_wait_before_first_collect = 0
    #iters_per_adjust = 55000
    iters_per_adjust = (55000/batch_size)/2
    max_batch_size = 10000

    #iters_per_adjust = 50
    #iters_per_adjust = 1
    per_variable = False
    base_optimizer = 'SGD'

    #for adam
    beta1 = 0.9
    beta2 = 0.999

    #for Adadelta
    rho = 0.95

    weighted_batch=False
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()

@ex.automain
@LogFileWriter(ex)
def my_main(lr, batch_size, n_epochs, iters_per_adjust, per_variable, iters_to_wait_before_first_collect, max_batch_size,
            base_optimizer, beta1, beta2, rho, weighted_batch, tensorboard_dir):

    bp = MnistBatchProvider(batch_size, False)
    x, y = bp.batch()

    enable_dropout = tf.Variable(True, trainable=False)
    model = cnn_model_fn(x, y, enable_dropout, weighted_batch)

    train_accuracy_summary = tf.summary.scalar('train_accuracy', model.accuracy, ['mnist_summary'])
    test_accuracy_summary = tf.summary.scalar('test_accuracy', model.accuracy, ['mnist_summary'])
    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['mnist_summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['mnist_summary'])

    test_summary = tf.summary.merge([test_loss_summary, test_accuracy_summary])
    train_summary = tf.summary.merge([train_loss_summary, train_accuracy_summary])

    set_dropout = [tf.assign(enable_dropout, False), tf.assign(enable_dropout, True)]

    optimizer = SgdBatchAdjustOptimizer(model.loss, bp, tf.trainable_variables(),
                                 lr=lr,
                                 batch_size=batch_size,
                                 train_dataset_size=bp.train_size(),
                                 iters_per_adjust=iters_per_adjust,
                                 per_variable=per_variable,
                                 max_batch_size=max_batch_size,
                                 iters_to_wait_before_first_collect=iters_to_wait_before_first_collect,
                                 base_optimizer=base_optimizer,
                                 beta1=beta1,
                                 beta2=beta2,
                                 rho=rho)


    with tf.Session() as sess:

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
        writer.flush()

        optimizer.set_summary_writer(writer)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        bp.custom_runner.start_threads(sess, n_train_threads=2)

        for epoch in range(n_epochs):
            print 'epoch = ' + str(epoch)
            print 'Write graph into tensorboard into: ' + str(tensorboard_dir)
            ex.info['epoch'] = epoch

            print 'Calculating accuracy...'
            sess.run(set_dropout[0])

            bp.set_deque_batch_size(sess, bp.train_size()/2)
            writer.add_summary(sess.run(train_accuracy_summary, {model.accuracy: avarge_n_calls(sess, model.accuracy, 2)}), epoch)
            writer.add_summary(sess.run(train_loss_summary, {model.loss: avarge_n_calls(sess, model.loss, 2)}), epoch)

            bp.set_data_source(sess, 'test')
            bp.set_deque_batch_size(sess, bp.test_size())
            writer.add_summary(sess.run(test_summary), epoch)

            bp.set_data_source(sess, 'train')
            bp.set_deque_batch_size(sess, optimizer.batch_size)
            sess.run(set_dropout[1])

            #optimize
            image_procceses = 0
            while image_procceses < bp.train_size():
            #for i in range(bp.train_size()/optimizer.batch_size):
                image_procceses += optimizer.batch_size
                optimizer.run_iter(sess)
            writer.flush()
            print '---------------'