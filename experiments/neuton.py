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

def golden_func(t):
    return 0.5*9.81*(t**2)

def _model(features, layers_size, weight_decay):
    # Input Layer
    layers = [features]

    for i, size in zip(range(len(layers_size)), layers_size):
        if i < len(layers_size) - 1:
            layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=tf.nn.tanh, use_bias=True))
        else:
            layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=None, use_bias=True))

    logits = layers[-1]

    loss = tf.losses.mean_squared_error(logits, golden_func(features))

    costs = []
    for var in tf.trainable_variables():
        if not (var.op.name.find(r'bias') > 0):
            costs.append(tf.nn.l2_loss(var))

    loss += tf.multiply(weight_decay, tf.add_n(costs))

    return Model(loss=loss, accuracy=None, predictions=logits)


from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('neuton_law')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='neuton_law_db'))


@ex.config
def my_config():

    lr = 1e-4
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

    layers_size = [100, 1]

    weighted_batch = False
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()
    fashion_mnist = True
    weight_decay = 0.0

    step_size_anealing = False

@ex.automain
@LogFileWriter(ex)
def my_main(lr, batch_size, n_epochs, iters_per_adjust, per_variable, iters_to_wait_before_first_collect, weight_decay, layers_size,
            lr_update_formula_risky, step_size_anealing,
				base_optimizer, beta1, beta2, rho, weighted_batch, tensorboard_dir, fashion_mnist, seed):


    x = tf.random_uniform(
        [batch_size, 1],
        minval= -10.0,
        maxval= 10.0,
        dtype=tf.float32)

    model = _model(x, layers_size, weight_decay)

    hess = tf.gradients(tf.gradients(model.predictions, x)[0][0], x)[0][0]


    train_loss_summary = tf.summary.scalar('train_loss', model.loss, ['mnist_summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss, ['mnist_summary'])

    test_summary = tf.summary.merge([test_loss_summary])
    train_summary = tf.summary.merge([train_loss_summary])

    optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    optim_op = optimizer.minimize(model.loss)

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
        writer.flush()

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())


        for epoch in range(n_epochs):
            print 'epoch = ' + str(epoch)
            print 'Write graph into tensorboard into: ' + str(tensorboard_dir)
            ex.info['epoch'] = epoch

            print 'Calculating accuracy...'
            acc = avarge_n_calls(sess, model.loss, 1000)
            print 'loss = ' + str(acc)

            print 'hess = ' + str(avarge_n_calls(sess, hess, 1000))

            writer.add_summary(sess.run(train_loss_summary, {model.loss: acc}), epoch)

            #optimize
            for i in range(1000):
                sess.run(optim_op)
            writer.flush()
            print '---------------'