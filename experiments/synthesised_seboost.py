# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from seboost_optimizer import SeboostOptimizer
from batch_provider import SimpleBatchProvider

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'predictions'])


def cnn_model_fn(features, labels, input_dim):
  """Model function for CNN."""
  # Input Layer
  labels = tf.reshape(labels, [-1])

  layer1 = tf.layers.dense(inputs=features, units=input_dim, activation=tf.nn.tanh)
  layer2 = tf.layers.dense(inputs=layer1, units=input_dim, activation=tf.nn.tanh)
  layer3 = tf.layers.dense(inputs=layer2, units=input_dim, activation=tf.nn.tanh)
  layer4 = tf.layers.dense(inputs=layer3, units=input_dim, activation=tf.nn.tanh)
  logits = tf.layers.dense(inputs=layer4, units=1)


  loss_per_sample = tf.squared_difference(tf.reshape(logits, [-1]), labels, name='loss_per_sample')
  loss = tf.reduce_mean(loss_per_sample, name='loss')


  return Model(loss=loss, predictions=logits)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('synthesised_seboost')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='synthesised_seboost_db'))

@ex.config
def my_config():
    lr = 0.1
    n_epochs = 30
    batch_size = 10
    history_size = 10
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()


    VECTOR_BREAKING = False
    adaptable_learning_rate = False
    num_of_batches_per_sesop = 10

    disable_dropout_during_sesop = True
    sesop_method = 'CG'
    #sesop_method = 'natural_gradient'
    sesop_options = {'maxiter': 200, 'gtol': 1e-5}
    seboost_base_method = 'SGD'

    normalize_function_during_sesop = True
    history_decay_rate = 0.0

    seed = 913526365
    input_dim = 6
    dont_run_sesop = False
    dataset_size = 5000

    sesop_private_dataset = True
    if dont_run_sesop == False and sesop_private_dataset == True:
        sesop_dataset_size = dataset_size/10
    else:
        sesop_dataset_size = 0

    # anchor_size = 2
    # anchor_offsets = range(30)

    anchor_size = 2
    anchor_offsets = [1, 15]

@ex.automain
@LogFileWriter(ex)
def my_main(lr, VECTOR_BREAKING, history_size, adaptable_learning_rate, batch_size, anchor_size, anchor_offsets,
            num_of_batches_per_sesop, sesop_private_dataset, n_epochs, disable_dropout_during_sesop,history_decay_rate,
            sesop_method, sesop_options, seboost_base_method, normalize_function_during_sesop,
            dataset_size, input_dim, dont_run_sesop, sesop_dataset_size, tensorboard_dir):

    bp = SimpleBatchProvider(input_dim=input_dim, output_dim=1, dataset_size=dataset_size, batch_size=batch_size, sesop_dataset_size=sesop_dataset_size)
    x, y = bp.batch()

    model = cnn_model_fn(x, y, input_dim)

    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['summary'])

    test_summary = tf.summary.merge([test_loss_summary])
    train_summary = tf.summary.merge([train_loss_summary])


    optimizer = SeboostOptimizer(model.loss, bp, tf.trainable_variables(), history_size=history_size,
                                 VECTOR_BREAKING=VECTOR_BREAKING,
                                 lr=lr,
                                 batch_size=batch_size,
                                 train_dataset_size=bp.train_size(),
                                 adaptable_learning_rate=adaptable_learning_rate,
                                 num_of_batches_per_sesop=num_of_batches_per_sesop,
                                 history_decay_rate=history_decay_rate,
                                 sesop_private_dataset=sesop_private_dataset,
                                 seboost_base_method=seboost_base_method,
                                 anchor_size=anchor_size,
                                 anchor_offsets=anchor_offsets,
                                 sesop_method=sesop_method,
                                 predictions=model.predictions,
                                 normalize_function_during_sesop=normalize_function_during_sesop,
                                 sesop_options=sesop_options)

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:

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

            print 'Calculating loss...'

            bp.set_deque_batch_size(sess, bp.train_size()/2)
            writer.add_summary(sess.run(train_loss_summary, {model.loss: avarge_n_calls(sess, model.loss, 2)}), epoch)

            bp.set_data_source(sess, 'test')
            bp.set_deque_batch_size(sess, bp.test_size())
            writer.add_summary(sess.run(test_summary), epoch)

            bp.set_data_source(sess, 'train')
            bp.set_deque_batch_size(sess, batch_size)

            #optimize
            optimizer.run_epoch(sess)

            #optimizer.iter += 1
            if dont_run_sesop == True:
                optimizer.iter += 1
            else:
                optimizer.run_sesop(sess)



            print '---------------'