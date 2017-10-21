# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from seboost_optimizer import SeboostOptimizer


from batch_provider import MnistBatchProvider, FashionMnistBatchProvider
from resnet_model_original import ResNet, HParams

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'accuracy', 'predictions'])

def cnn_model_fn(features, labels, enable_dropout):
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
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)

  # costs = []
  # for var in tf.trainable_variables():
  #     if not (var.op.name.find(r'bias') > 0):
  #         costs.append(tf.nn.l2_loss(var))
  #
  # loss += tf.multiply(weight_decay, tf.add_n(costs))

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return Model(loss=loss, accuracy=accuracy, predictions=logits)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('minst_seboost')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il', db_name='minst_seboost_db'))

@ex.config
def my_config():
    lr = 0.1
    n_epochs = 100
    batch_size = 100
    history_size = 10
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()


    VECTOR_BREAKING = False
    adaptable_learning_rate = False
    num_of_batches_per_sesop = 10
    sesop_private_dataset = False
    disable_dropout_during_sesop = True
    disable_dropout_at_all = True
    disable_sesop_at_all = False

    if disable_sesop_at_all:
        sesop_private_dataset = False

    sesop_method = 'CG'
    #sesop_method = 'natural_gradient'
    sesop_options = {'maxiter': 200, 'gtol': 1e-6}
    seboost_base_method = 'SGD'

    normalize_function_during_sesop = True

    weight_decay = 0.0000

    history_decay_rate = 1.0

    per_variable = False
    anchor_size = 0
    anchor_offsets = []
    iters_per_sesop = 550 #once per epoch
    use_grad_dir = False
    normalize_history_dirs = False
    normalize_subspace = False
    break_sesop_batch = False
    seed = 913526365

    fashion_mnist = False
    model = 'cnn'

    #SgdAdjustOptimizer
@ex.automain
@LogFileWriter(ex)
def my_main(lr, weight_decay, VECTOR_BREAKING, history_size, adaptable_learning_rate, batch_size, per_variable, anchor_size, anchor_offsets, iters_per_sesop,
            use_grad_dir, normalize_history_dirs, normalize_subspace, break_sesop_batch, fashion_mnist, model,
            num_of_batches_per_sesop, sesop_private_dataset, n_epochs, disable_dropout_during_sesop, seed,
            sesop_method, sesop_options, seboost_base_method, normalize_function_during_sesop, disable_dropout_at_all, disable_sesop_at_all, history_decay_rate,
            tensorboard_dir):

    if fashion_mnist == False:
        bp = MnistBatchProvider(batch_size, sesop_private_dataset, seed)
    else:
        bp = FashionMnistBatchProvider(batch_size, sesop_private_dataset, seed)

    x, y = bp.batch()

    enable_dropout = tf.Variable(True, trainable=False)
    # model = cnn_model_fn(x, y, enable_dropout)
    if model == 'cnn':
        model = cnn_model_fn(x, y, enable_dropout)
    elif model == 'wide-resnet':

        hps = HParams(batch_size=batch_size,
                      num_classes=10,
                      min_lrn_rate=None,
                      lrn_rate=None,
                      num_residual_units=4,
                      use_bottleneck=False,
                      weight_decay_rate=weight_decay,
                      relu_leakiness=0.1,
                      state_of_the_art=False,
                      optimizer=None,
                      input_chanels=1)

        # onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)
        onehot_labels = tf.one_hot(indices=tf.cast(tf.reshape(y, [-1]), tf.int32), depth=10)

        model = ResNet(hps, tf.reshape(x, [-1, 28, 28, 1]), onehot_labels, 'train')
        model._build_model()
        model = Model(loss=model.cost, accuracy=model.accuracy, predictions=None)

    train_accuracy_summary = tf.summary.scalar('train_accuracy', model.accuracy, ['mnist_summary'])
    test_accuracy_summary = tf.summary.scalar('test_accuracy', model.accuracy, ['mnist_summary'])
    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['mnist_summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['mnist_summary'])

    test_summary = tf.summary.merge([test_loss_summary, test_accuracy_summary])
    train_summary = tf.summary.merge([train_loss_summary, train_accuracy_summary])

    set_dropout = [tf.assign(enable_dropout, False), tf.assign(enable_dropout, True)]

    optimizer = SeboostOptimizer(model.loss, bp, tf.trainable_variables(), history_size=history_size,
                                 VECTOR_BREAKING=VECTOR_BREAKING,
                                 lr=lr,
                                 batch_size=batch_size,
                                 train_dataset_size=bp.train_size(),
                                 adaptable_learning_rate=adaptable_learning_rate,
                                 num_of_batches_per_sesop=num_of_batches_per_sesop,
                                 sesop_private_dataset=sesop_private_dataset,
                                 seboost_base_method=seboost_base_method,
                                 history_decay_rate=history_decay_rate,
                                 sesop_method=sesop_method,
                                 disable_sesop_at_all=disable_sesop_at_all,
                                 predictions=model.predictions,
                                 normalize_history_dirs=normalize_history_dirs,
                                 break_sesop_batch=break_sesop_batch,
                                 use_grad_dir=use_grad_dir,
                                 normalize_subspace=normalize_subspace,
                                 anchor_size=anchor_size,
                                 iters_per_sesop=iters_per_sesop,
                                 anchor_offsets=anchor_offsets,
                                 normalize_function_during_sesop=normalize_function_during_sesop,
                                 weight_decay=weight_decay,
                                 per_variable=per_variable,
                                 sesop_options=sesop_options)


    with tf.Session() as sess:

        #shutil.rmtree('/tmp/generated_data/1')



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
            bp.set_deque_batch_size(sess, batch_size)
            sess.run(set_dropout[1])

            #optimize
            if disable_dropout_at_all == True:
                sess.run(set_dropout[0])

            for i in range(bp.train_size() / batch_size):
                optimizer.run_iter_without_sesop(sess)

            if disable_dropout_during_sesop == True:
                sess.run(set_dropout[0])

            optimizer.run_sesop(sess)

            if disable_dropout_during_sesop == True and disable_dropout_at_all == False:
                sess.run(set_dropout[1])

            writer.flush()
            print '---------------'