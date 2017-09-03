# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from alternative_optimizer import AlternativeOptimizer
from batch_provider import PhysicsBatchProvider

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'predictions'])


def cnn_model_fn(features, labels, layers_size, weight_decay):
  """Model function for CNN."""
  # Input Layer
  labels = tf.reshape(labels, [-1])

  layers_size.append(1)

  layers = [features]
  for size in layers_size:
      layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=tf.nn.tanh))

  logits = layers[-1]

  loss_per_sample = tf.squared_difference(tf.reshape(logits, [-1]), labels, name='loss_per_sample')
  loss = tf.reduce_mean(loss_per_sample, name='loss')

  costs = []
  for var in tf.trainable_variables():
      if not (var.op.name.find(r'bias') > 0):
          costs.append(tf.nn.l2_loss(var))

  loss += tf.multiply(weight_decay, tf.add_n(costs))


  return Model(loss=loss, predictions=logits)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('physics_alternative')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='physics_alternative_db'))

@ex.config
def my_config():
    lr_for_main = 0.5
    lr_for_lr = None
    run_baseline = False
    main_base_method = 'SGD'
    lr_base_method = 'SGD'

    n_epochs = 200
    batch_size = 100

    momentum = 0.9
    # for adam
    beta1 = 0.9
    beta2 = 0.999

    # for Adadelta
    rho = 0.95

    tensorboard_dir = tf_utils.allocate_tensorboard_dir()

    seed = 913526365
    layers_size = [12, 8, 4]
    weight_decay = 0.0001


# optimizer = \
#     ScipyOptimizerInterface(loss=model.loss, var_list=tf.trainable_variables(), \
#                             equalities=None, method='BFGS', options=sesop_options)

@ex.automain
@LogFileWriter(ex)
def my_main(lr_for_main, lr_for_lr, run_baseline, main_base_method, lr_base_method, n_epochs, batch_size, momentum, weight_decay,
            beta1, beta2, rho, layers_size, tensorboard_dir, seed):

    bp = PhysicsBatchProvider(batch_size=batch_size, test_dataset_size=2000, sesop_dataset_size=0)
    x, y = bp.batch()

    model = cnn_model_fn(x, y, layers_size, weight_decay)

    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['summary'])

    test_summary = tf.summary.merge([test_loss_summary])
    train_summary = tf.summary.merge([train_loss_summary])


    optimizer = AlternativeOptimizer(model.loss, tf.trainable_variables(),
                                     lr_for_main=lr_for_main,
                                     lr_for_lr=lr_for_lr,
                                     run_baseline=run_baseline,
                                     main_base_method=main_base_method,
                                     lr_base_method=lr_base_method,
                                     batch_size=batch_size,
                                     beta1=beta1,
                                     beta2=beta2,
                                     rho=rho,
                                     momentum=momentum)

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
            for i in range(bp.train_size() / batch_size):
                optimizer.minimize(session=sess)

            print '---------------'