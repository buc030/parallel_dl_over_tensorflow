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

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'predictions'])

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

    return Model(loss=loss, predictions=None)



from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('minst_autoencoder_seboost')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il', db_name='minst_autoencoder_seboost_db'))

@ex.config
def my_config():

    layers_size = [28 * 28, 40, 20, 40, 28 * 28]


    lr = 0.1


    history_size = 10
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()


    VECTOR_BREAKING = False
    adaptable_learning_rate = True
    num_of_batches_per_sesop = 10
    sesop_private_dataset = False
    disable_dropout_during_sesop = True
    disable_dropout_at_all = True
    disable_sesop_at_all = False
    normalize_history_dirs = False
    normalize_subspace = False
    per_variable = False
    use_grad_dir = False

    if disable_sesop_at_all:
        sesop_private_dataset = False

    sesop_method = 'CG'
    sesop_options = {'maxiter': 200, 'gtol': 1e-06}
    #adaptable_learning_rate = False
    seboost_base_method = 'SGD'
    #seboost_base_method = 'SGD'

    normalize_function_during_sesop = True

    weight_decay = 0.0000

    history_decay_rate = 0.0

    seed = 913526365

    # anchor_size = 2
    # anchor_offsets = [1, 15]
    anchor_size = 0
    anchor_offsets = []


    # for adam
    beta1 = 0.9
    beta2 = 0.999

    # for Adadelta
    rho = 0.95

    batch_size = 100
    fashion_mnist = True
    iters_per_adjust = 1000
    base_optimizer = 'SGD'
    redo_tag = True

    n_epochs = 50

@ex.automain
@LogFileWriter(ex)
def my_main(lr, weight_decay, VECTOR_BREAKING, history_size, adaptable_learning_rate, batch_size, anchor_size, anchor_offsets, normalize_history_dirs, normalize_subspace, per_variable,
            num_of_batches_per_sesop, sesop_private_dataset, n_epochs, disable_dropout_during_sesop, beta1, beta2, rho, seed, use_grad_dir, layers_size, fashion_mnist,
            sesop_method, sesop_options, seboost_base_method, normalize_function_during_sesop, disable_dropout_at_all, disable_sesop_at_all, history_decay_rate,
            iters_per_adjust, base_optimizer, redo_tag,
            tensorboard_dir):

    #initial_batch_size, sesop_private_dataset, seed, mess_with_data
    if fashion_mnist == False:
        bp = MnistBatchProvider(batch_size, sesop_private_dataset, seed)
    else:
        bp = FashionMnistBatchProvider(batch_size, sesop_private_dataset, seed)

    x, y = bp.batch()

    model = autoencoder_model(x, layers_size, weight_decay)

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
                                 sesop_options=sesop_options,
                                 weight_decay=weight_decay,
                                 use_grad_dir=use_grad_dir,
                                 break_sesop_batch=False,
                                 iters_per_adjust=iters_per_adjust,
                                 base_optimizer=base_optimizer)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
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

            print 'Training ...'
            # optimize
            for i in range(bp.train_size() / batch_size):
                optimizer.run_iter_without_sesop(sess)

            optimizer.run_sesop(sess)


            writer.flush()
            print '---------------'