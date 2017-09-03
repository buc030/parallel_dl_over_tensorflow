# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from seboost_optimizer import SeboostOptimizer
from batch_provider import MnistBatchProvider

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'predictions'])


def autoencoder_model(features, labels, enable_preceptgene):
  """Model function for CNN."""
  # Input Layer
  input_layer = features

  #input_layer = tf.log(input_layer)

  if enable_preceptgene == False:
      #hidden1 = tf.layers.dense(inputs=input_layer, units=20, activation=tf.nn.softplus)

      hidden1 = tf.nn.softplus(tf.layers.batch_normalization(tf.layers.dense(inputs=input_layer, units=20, activation=None)))

      n_hidden = 4
      hiddens = [hidden1]
      for i in range(n_hidden - 1):
          hiddens.append(tf.nn.softplus(
              tf.layers.dense(inputs=hiddens[-1], units=40, activation=None)))

          #hiddens.append(tf.nn.softplus(tf.layers.batch_normalization(tf.layers.dense(inputs=hiddens[-1], units=40, activation=None))))


      output = tf.layers.dense(inputs=hiddens[-1], units=28*28, activation=None)


      loss = tf.losses.mean_squared_error(input_layer, output)
  else:

      input_layer = tf.log(((input_layer + 1)/2)*2.719)
      hidden1 = tf.layers.dense(inputs=input_layer, units=20, activation=None)

      hidden1 = tf.maximum(hidden1, 1.0)
      hidden1 = tf.log(hidden1)
      hidden2 = tf.layers.dense(inputs=hidden1, units=40, activation=None)

      hidden2 = tf.maximum(hidden2, 1.0)
      hidden2 = tf.log(hidden2)
      output = tf.layers.dense(inputs=hidden2, units=28 * 28, activation=None)

      loss = tf.losses.mean_squared_error(input_layer, output)


  return Model(loss=loss, predictions=output)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('minst_autoencoder_seboost')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='minst_autoencoder_seboost_db'))

@ex.config
def my_config():
    lr = 0.1
    n_epochs = 100
    batch_size = 100
    history_size = 10
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()


    VECTOR_BREAKING = False
    adaptable_learning_rate = True
    num_of_batches_per_sesop = 10
    sesop_private_dataset = True
    disable_dropout_during_sesop = True
    disable_dropout_at_all = True
    disable_sesop_at_all = False
    normalize_history_dirs = True
    normalize_subspace = True
    per_variable = True

    if disable_sesop_at_all:
        sesop_private_dataset = False

    sesop_options = {'maxiter': 10, 'gtol': 1e-10}
    #adaptable_learning_rate = False
    seboost_base_method = 'Momentum'
    #seboost_base_method = 'SGD'

    normalize_function_during_sesop = True

    weight_decay = 0.0000

    history_decay_rate = 0.0

    enable_preceptgene = False
    seed = 913526365

    anchor_size = 2
    anchor_offsets = [1, 15]


    # for adam
    beta1 = 0.9
    beta2 = 0.999

    # for Adadelta
    rho = 0.95



@ex.automain
@LogFileWriter(ex)
def my_main(lr, weight_decay, VECTOR_BREAKING, history_size, adaptable_learning_rate, batch_size, anchor_size, anchor_offsets, normalize_history_dirs, normalize_subspace, per_variable,
            num_of_batches_per_sesop, sesop_private_dataset, n_epochs, disable_dropout_during_sesop, enable_preceptgene, beta1, beta2, rho,
            sesop_method, sesop_options, seboost_base_method, normalize_function_during_sesop, disable_dropout_at_all, disable_sesop_at_all, history_decay_rate,
            tensorboard_dir):

    bp = MnistBatchProvider(batch_size, sesop_private_dataset)
    x, y = bp.batch()

    model = autoencoder_model(x, y, enable_preceptgene)

    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['mnist_summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['mnist_summary'])

    test_summary = tf.summary.merge([test_loss_summary])
    train_summary = tf.summary.merge([train_loss_summary])


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
                                 anchor_size=anchor_size,
                                 disable_sesop_at_all=disable_sesop_at_all,
                                 per_variable=per_variable,
                                 normalize_history_dirs=normalize_history_dirs,
                                 normalize_subspace=normalize_subspace,
                                 beta1=beta1,
                                 beta2=beta2,
                                 rho=rho,
                                 anchor_offsets=anchor_offsets,
                                 predictions=model.predictions,
                                 normalize_function_during_sesop=normalize_function_during_sesop,
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


            #THIS IS LOGING
            print 'Calculating accuracy...'
            bp.set_deque_batch_size(sess, bp.train_size()/2)
            writer.add_summary(sess.run(train_loss_summary, {model.loss: avarge_n_calls(sess, model.loss, 2)}), epoch)

            bp.set_data_source(sess, 'test')
            bp.set_deque_batch_size(sess, bp.test_size())
            writer.add_summary(sess.run(test_summary), epoch)

            bp.set_data_source(sess, 'train')
            bp.set_deque_batch_size(sess, batch_size)
            #END LOGING


            #optimize
            optimizer.run_epoch(sess)

            if disable_sesop_at_all == True:
                optimizer.iter += 1
            else:
                optimizer.run_sesop(sess)

            writer.flush()
            print '---------------'