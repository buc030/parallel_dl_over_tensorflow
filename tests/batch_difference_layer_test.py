# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn import SKCompat

from batch_difference_layer import BatchDifferenceLayer
from tensorflow.examples.tutorials.mnist import input_data
import shutil
import tf_utils

def my_model(features, labels, is_training):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 28*28])

  layers = [BatchDifferenceLayer(input=input_layer, labels=labels, n_in=28*28, n_out=2500, n_labels=10, is_training=is_training, activation=tf.nn.relu)]

  for size in [(2500, 2000), (2000, 1500), (1500, 1000), (1000, 500)]:
      layers.append(BatchDifferenceLayer(input=layers[-1].out, labels=labels, n_in=size[0], n_out=size[1], n_labels=10, is_training=is_training, activation=tf.nn.relu))

  logits = tf.layers.dense(inputs=layers[-1].out, units=10)


  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sgd_optim = tf.train.GradientDescentOptimizer(0.1, use_locking=False)
  sgd_op = sgd_optim.minimize(loss)
  sgd_grads = tf.gradients(loss, tf.trainable_variables())
  return loss, sgd_op, accuracy, sgd_grads, layers


def cnn_model_fn(features, labels):
  """Model function for CNN."""
  # Input Layer
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
      inputs=dense, rate=0.4, training=True)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)


  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sgd_optim = tf.train.GradientDescentOptimizer(0.001)
  sgd_op = sgd_optim.minimize(loss)
  return loss, sgd_op, accuracy

validation_size = 5000
train_size = 60000 - validation_size

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, validation_size=validation_size)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None])

#loss, sgd_op, accuracy = cnn_model_fn(x, y_)

is_training = tf.placeholder(tf.bool)
loss, sgd_op, accuracy, sgd_grads, layers = my_model(x, y_, is_training)

with tf.Session() as sess:

    #shutil.rmtree('/tmp/generated_data/1')
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()

    print 'Write graph into tensorboard into: ' + str(tensorboard_dir)
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(sess.graph)
    writer.flush()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        print 'epoch = ' + str(epoch)



        batch = mnist.train.next_batch(5000)
        print 'loss = ' + str(sess.run(loss, feed_dict={x: batch[0], y_: batch[1], is_training: False}))
        print 'accuracy = ' + str(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], is_training: False}))

        batch = mnist.test.next_batch(5000)
        print 'test accuracy = ' + str(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], is_training: False}))

        batch_size = 100
        for _ in range(train_size / batch_size):
            batch = mnist.train.next_batch(batch_size)
            for i in range(1):
                # print 'W b4 = ' + str(sess.run(layer1.W)[0][0])
                # print 'loss b4 = ' + str(sess.run(loss, feed_dict={x: batch[0], y_: batch[1], is_training: True}))
                #
                #
                # #print 'grad W = ' + str(sess.run(sgd_grads, feed_dict={x: batch[0], y_: batch[1], is_training: True}))
                # print 'grad W norm = ' + str(sess.run(tf.global_norm(sgd_grads), feed_dict={x: batch[0], y_: batch[1], is_training: True}))
                # print 'W after = ' + str(sess.run(layer1.W)[0][0])
                _loss, _ = sess.run([loss, sgd_op],
                                    feed_dict={x: batch[0], y_: batch[1], is_training: True})
                #print 'loss after = ' + str(sess.run(loss, feed_dict={x: batch[0], y_: batch[1], is_training: True}))