# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from lr_auto_adjust_sgd import SgdAdjustOptimizer
from batch_provider import MnistBatchProvider, FashionMnistBatchProvider

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'accuracy', 'predictions'])

def autoencoder_model(features, layers_size, weight_decay):
    # Input Layer
    features = tf.reshape(features, [-1, 28*28])

    layers = [features]

    for i, size in zip(range(len(layers_size)), layers_size):
        layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=tf.nn.tanh, use_bias=True))

    logits = layers[-1]

    loss = tf.losses.mean_squared_error(features, logits)

    costs = []
    for var in tf.trainable_variables():
        if not (var.op.name.find(r'bias') > 0):
            costs.append(tf.nn.l2_loss(var))

    loss += tf.multiply(weight_decay, tf.add_n(costs))

    return Model(loss=loss, accuracy=None, predictions=None)


from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('minst_autoencoder_sgd_adjust')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il',db_name='minst_autoencoder_sgd_adjust_db'))


@ex.config
def my_config():

    lr = 0.001
    batch_size = 100
    n_epochs = 100
    lr_update_formula_risky = True

    iters_to_wait_before_first_collect = 0
    iters_per_adjust = 55000
    #iters_per_adjust = (55000/batch_size)/2
    #iters_per_adjust = 50
    #iters_per_adjust = 1
    per_variable = False
    base_optimizer = 'SGD'

    seed = 913526365
    #for adam
    beta1 = 0.9
    beta2 = 0.999

    #for Adadelta
    rho = 0.95

    layers_size = [28*28, 1000, 2000, 1000, 28*28]

    weighted_batch = False
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()
    fashion_mnist = True
    weight_decay = 0.0002

    step_size_anealing = False



@ex.automain
@LogFileWriter(ex)
def my_main(lr, batch_size, n_epochs, iters_per_adjust, per_variable, iters_to_wait_before_first_collect, weight_decay, layers_size,
            lr_update_formula_risky, step_size_anealing,
				base_optimizer, beta1, beta2, rho, weighted_batch, tensorboard_dir, fashion_mnist, seed):

    if fashion_mnist == False:
        bp = MnistBatchProvider(batch_size, False, seed)
    else:
        bp = FashionMnistBatchProvider(batch_size, False, seed)

    x, y = bp.batch()


    model = autoencoder_model(x, layers_size, weight_decay)

    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['mnist_summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['mnist_summary'])

    test_summary = tf.summary.merge([test_loss_summary])
    train_summary = tf.summary.merge([train_loss_summary])

    optimizer = SgdAdjustOptimizer(model.loss, bp, tf.trainable_variables(),
                                 lr=lr,
                                 batch_size=batch_size,
                                 train_dataset_size=bp.train_size(),
                                 iters_per_adjust=iters_per_adjust,
                                 per_variable=per_variable,
                                 iters_to_wait_before_first_collect=iters_to_wait_before_first_collect,
                                 base_optimizer=base_optimizer,
                                 beta1=beta1,
                                 beta2=beta2,
                                 rho=rho,
                                 lr_update_formula_risky=lr_update_formula_risky,
                                   step_size_anealing=step_size_anealing)


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
            bp.set_deque_batch_size(sess, bp.train_size()/2)
            writer.add_summary(sess.run(train_loss_summary, {model.loss: avarge_n_calls(sess, model.loss, 2)}), epoch)

            bp.set_data_source(sess, 'test')
            bp.set_deque_batch_size(sess, bp.test_size())
            writer.add_summary(sess.run(test_summary), epoch)

            bp.set_data_source(sess, 'train')
            bp.set_deque_batch_size(sess, batch_size)

            #optimize
            for i in range(bp.train_size()/batch_size):
                optimizer.run_iter(sess)
            writer.flush()
            print '---------------'