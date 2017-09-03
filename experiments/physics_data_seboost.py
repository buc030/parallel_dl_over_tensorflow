# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils
import random
import scipy
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from seboost_optimizer import SeboostOptimizer
from batch_provider import PhysicsBatchProvider
from my_external_optimizer import ScipyOptimizerInterface

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

Model = collections.namedtuple('Model', ['loss', 'loss_test_full', 'loss_train_full', 'predictions'])

from tensorflow.python.ops.init_ops import Initializer, _compute_fans, VarianceScaling
from tensorflow.python.framework import dtypes

class IntersectionInit(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        scaleing = VarianceScaling(1)
        return scaleing(shape)

    def get_config(self):
        return {"dtype": self.dtype.name}


class IntersectionInitBias(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        scale_shape = shape
        if partition_info is not None:
            scale_shape = partition_info.full_shape
        fan_in, fan_out = _compute_fans(scale_shape)

        scaleing = VarianceScaling(fan_in)
        return scaleing(shape)

    def get_config(self):
        return {"dtype": self.dtype.name}


def cnn_model_fn(features, labels, layers_size, test_dataset_size):
  """Model function for CNN."""
  # Input Layer
  labels = tf.reshape(labels, [-1])

  layers_size.append(1)

  layers = [features]

  for i, size in zip(range(len(layers_size)), layers_size):



      #
      # _activation = tf.nn.tanh
      # if i < len(layers_size) - 1:
      #     #y = wx
      #     #the goal of m is to put the grad of wx which is x close to 1
      #     # xm = 1
      #     _activation = lambda y: ((y*m)/(tf.abs(y))) #+ 1e-10))
      #     #layers.append ((layers[-1]*tf.reduce_mean(layers[-2]))/(tf.abs(layers[-1]) + 1e-10))
      #     layers.append((layers[-1] * tf.Variable(np.ones(1, dtype=np.float32))) / (tf.abs(layers[-1]) ))
      # else:
      #
      #layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=tf.nn.tanh, kernel_initializer=IntersectionInit(), bias_initializer=IntersectionInitBias()))
      layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=tf.nn.tanh, use_bias=True))

      #We want b to choose n/2 items out of n, so b should be about w^Tx
      # layers.append(layers[-1] - tf.reduce_sum(layers[-1], axis=0)/2)
      #
      # #TODO: try represting the ver as r.v. Each var is w/normal(0, mu(w))
      # #layers.append(tf.layers.dense(inputs=layers[-1], units=size, activation=tf.nn.tanh))
      # layers.append(tf.nn.tanh(layers[-1]))

  logits = layers[-1]

  loss_per_sample = tf.squared_difference(tf.reshape(logits, [-1]), labels, name='loss_per_sample')
  #loss = tf.reduce_mean(loss_per_sample, name='loss')
  loss = tf.reduce_sum(loss_per_sample, name='loss')




  return Model(loss=(loss)/tf.cast(tf.shape(loss_per_sample)[0], tf.float32) ,
               loss_test_full=loss/test_dataset_size, loss_train_full=loss/(20000 - test_dataset_size), predictions=logits)




from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('physics_seboost')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='physics_seboost_db'))

@ex.config
def my_config():
    lr = 0.5
    n_epochs = 200
    batch_size = 2000


    tensorboard_dir = tf_utils.allocate_tensorboard_dir()



    #SESOP options:

    disable_dropout_during_sesop = True
    sesop_method = 'BFGS'
    #sesop_method = 'natural_gradient'
    #sesop_options = {'maxiter': 200, 'maxfev' : 200}
    #sesop_options = {'maxiter': 200, 'gtol': 1e-10}
    sesop_options = {'maxiter': 10, 'gtol': 1e-10}
    adaptable_learning_rate = False
    seboost_base_method = 'Momentum'
    per_variable = False
    normalize_history_dirs = False
    normalize_subspace= False
    normalize_function_during_sesop = True
    use_grad_dir = True
    iters_per_sesop = 90
    history_decay_rate = 0.0
    anchor_size = 0
    anchor_offsets = []
    VECTOR_BREAKING = False
    history_size = 1
    num_of_batches_per_sesop = 1



    seed = 913526365
    input_dim = 6
    dont_run_sesop = False
    dataset_size = 20000
    test_dataset_size = 18000

    sesop_private_dataset = False
    if dont_run_sesop == False and sesop_private_dataset == True:
        sesop_dataset_size = 1000
    else:
        sesop_dataset_size = 0

    # anchor_size = 2
    # anchor_offsets = range(30)


    momentum = 0.9

    layers_size = [12, 8, 4]
    #weight_decay = 0.0001
    weight_decay = 0.000001

    #for performance
    break_sesop_batch = True

    # for adam
    beta1 = 0.9
    beta2 = 0.999

    # for Adadelta
    rho = 0.95

    initial_accumulator_value = 0.1

    debug_pure_CG = True


# optimizer = \
#     ScipyOptimizerInterface(loss=model.loss, var_list=tf.trainable_variables(), \
#                             equalities=None, method='BFGS', options=sesop_options)

@ex.automain
@LogFileWriter(ex)
def my_main(lr, VECTOR_BREAKING, history_size, adaptable_learning_rate, batch_size, anchor_size, anchor_offsets, per_variable, layers_size, momentum, weight_decay,
            beta1, beta2, rho, break_sesop_batch, seed, initial_accumulator_value, use_grad_dir, iters_per_sesop, debug_pure_CG,
            num_of_batches_per_sesop, sesop_private_dataset, n_epochs, disable_dropout_during_sesop,history_decay_rate, normalize_history_dirs, normalize_subspace,
            sesop_method, sesop_options, seboost_base_method, normalize_function_during_sesop, test_dataset_size,
            dataset_size, input_dim, dont_run_sesop, sesop_dataset_size, tensorboard_dir):

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    scipy.random.seed(seed)

    with tf.name_scope('batch_provider'):
        bp = PhysicsBatchProvider(batch_size=batch_size, test_dataset_size=test_dataset_size, sesop_dataset_size=sesop_dataset_size, seed=seed)
    x, y = bp.batch()

    with tf.name_scope('model'):
        model = cnn_model_fn(x, y, layers_size, test_dataset_size)

    train_loss_summary = tf.summary.scalar('train_loss', model.loss_train_full, ['summary'])
    test_loss_summary = tf.summary.scalar('test_loss', model.loss_test_full, ['summary'])

    test_summary = tf.summary.merge([test_loss_summary])
    train_summary = tf.summary.merge([train_loss_summary])

    if debug_pure_CG == False:
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
                                 use_grad_dir=use_grad_dir,
                                 anchor_offsets=anchor_offsets,
                                 sesop_method=sesop_method,
                                 iters_per_sesop=iters_per_sesop,
                                 weight_decay=weight_decay,
                                 disable_sesop_at_all=dont_run_sesop,
                                 beta1=beta1,
                                 beta2=beta2,
                                 rho=rho,
                                 break_sesop_batch=break_sesop_batch,
                                 initial_accumulator_value=initial_accumulator_value,
                                 predictions=model.predictions,
                                 per_variable=per_variable,
                                 momentum=momentum,
                                 normalize_history_dirs=normalize_history_dirs,
                                 normalize_subspace=normalize_subspace,
                                 normalize_function_during_sesop=normalize_function_during_sesop,
                                 sesop_options=sesop_options)
    else:
        optimizer = \
            ScipyOptimizerInterface(loss=model.loss, var_list=tf.trainable_variables(), \
                                    equalities=None, method=sesop_method, options={'maxiter': n_epochs, 'gtol': 1e-10})

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
        writer.flush()

        #SV TODO:
        #When we increase the batch size by factor m, the expectation of the gradient norm grows times m? thus lr has to change also.
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        bp.custom_runner.start_threads(sess, n_train_threads=1)

        if debug_pure_CG:
            optimizer.temp_iter = 0
            def callback(x):

                _loss, _grad = optimizer.loss_grad_func(x)
                s = sess.run(train_loss_summary, {model.loss_train_full : _loss})

                writer.add_summary(s, optimizer.temp_iter)
                optimizer.temp_iter += 1
                writer.flush()

            bp.set_deque_batch_size(sess, num_of_batches_per_sesop * batch_size)
            _x, _labels = sess.run([x, y])
            feed_dicts = [{x: _x, y: _labels}]
            bp.set_deque_batch_size(sess, batch_size)
            print 'running CG/BFGS...'
            optimizer.minimize(sess, feed_dicts, step_callback=callback)
            writer.flush()


        else:
            optimizer.set_summary_writer(writer)



            for epoch in range(n_epochs):
                print 'epoch = ' + str(epoch)
                print 'Write graph into tensorboard into: ' + str(tensorboard_dir)
                ex.info['epoch'] = epoch

                print 'Calculating loss...'

                bp.set_deque_batch_size(sess, bp.train_size())
                writer.add_summary(sess.run(train_loss_summary), epoch)

                bp.set_data_source(sess, 'test')
                bp.set_deque_batch_size(sess, bp.test_size())
                writer.add_summary(sess.run(test_summary), epoch)

                bp.set_data_source(sess, 'train')
                bp.set_deque_batch_size(sess, batch_size)


                #optimize
                print 'We have ' + str(bp.train_size() / batch_size) + ' iters to run this epcoh...'
                for i in range(bp.train_size() / batch_size):
                    optimizer.run_iter(sess)

                writer.flush()
            print '---------------'