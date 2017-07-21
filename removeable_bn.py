
import tensorflow as tf
import numpy as np

import os
from tensorflow.examples.tutorials.mnist import input_data

class RemoveableBNLayer:

    def __init__(self, _x, prev_layer_w, prev_layer_b, phase_train):
        self.x = _x
        self.is_bn_on = tf.Variable(True, trainable=False)
        self.set_is_bn_on = [tf.assign(self.is_bn_on, False), tf.assign(self.is_bn_on, True)]

        self.prev_layer_w = prev_layer_w
        self.prev_layer_b = prev_layer_b
        #self.x = tf.Print(_x, [prev_layer_b])
        self.eps = 0.000
        params_shape = [self.x.get_shape()[-1]]

        self.beta = tf.Variable(
            name='beta', dtype=tf.float32,
            initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=True)

        self.gamma = tf.Variable(
            name='gamma', dtype=tf.float32,
            initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=True)

        #fully connected
        if len(self.x.get_shape()) < 4:
            self.bn_batch_mean, self.bn_batch_variance = tf.nn.moments(self.x, [0], name='moments')
        else:
            #conv: BHWD (batch, h, w, depth)
            self.bn_batch_mean, self.bn_batch_variance = tf.nn.moments(self.x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        self.ema_apply_op = ema.apply([self.bn_batch_mean, self.bn_batch_variance])

        # for training with BN time
        def mean_var_with_update():
            #SV DEBUG
            #ema_apply_op = ema.apply([self.bn_batch_mean, self.bn_batch_variance])
            #with tf.control_dependencies([ema_apply_op]):
            return tf.identity(self.bn_batch_mean), tf.identity(self.bn_batch_variance)
                #return tf.Print(tf.identity(self.bn_batch_mean), [self.bn_batch_mean], 'update mean....'), tf.identity(self.bn_batch_variance)

        # for training without BN time, and for mode switch time
        def mean_var():
            return (tf.identity(self.bn_batch_mean), tf.identity(self.bn_batch_variance))


        # for testing time
        def moving_mean_var():
            res = (ema.average(self.bn_batch_mean), ema.average(self.bn_batch_variance))
            #if res[0] is None:
            #    return (tf.identity(self.bn_batch_mean), tf.identity(self.bn_batch_variance))
            return res




        #dont update moving mean while switching BN/NON-BN
        self.is_not_updating = tf.Variable(True, trainable=False)
        self.set_is_not_updating = [tf.assign(self.is_not_updating, False), tf.assign(self.is_not_updating, True)]


        cases = []
        #1. we take care of the case of while we are switching mode:
        self.mean, self.var = tf.cond(tf.logical_or(tf.logical_not(self.is_not_updating), tf.logical_not(self.is_bn_on)), mean_var,
                                      lambda: tf.cond(phase_train, mean_var_with_update, moving_mean_var))

        # #2. we are traning with BN
        # cases.append((tf.logical_and(phase_train, self.is_bn_on), mean_var_with_update))
        # #3. we are training without BN
        # default = moving_mean_var
        # self.mean, self.var = tf.case(pred_fn_pairs=cases, default=default)

        self.var.set_shape(self.bn_batch_variance.get_shape())
        self.mean.set_shape(self.bn_batch_mean.get_shape())

        #SV NOTE: uncomment this for switching to using moving mean.
        # self.mean, self.var = tf.cond(tf.logical_and(tf.logical_and(phase_train, self.is_bn_on), self.is_not_updating),
        #                     mean_var_with_update,
        #                     moving_mean_var)
                            #lambda: (ema.average(self.bn_batch_mean), ema.average(self.bn_batch_variance)))



        self.y_with_bn = tf.nn.batch_normalization(self.x, self.mean, self.var, self.beta, self.gamma, self.eps)
        self.y_without_bn = tf.identity(self.x)

        #self.out = tf.cast(self.is_bn_on, tf.float32)*self.y_with_bn + (1.0 - tf.cast(self.is_bn_on, tf.float32))*self.y_without_bn
        self.out = tf.cond(self.is_bn_on, lambda: self.y_with_bn, lambda: self.y_without_bn)


        self.out.set_shape(self.y_without_bn.get_shape())
        ################# SAVE values when transforming from BN to non BN ##############
        self.saved_var = tf.Variable(
            name='saved_var', dtype=tf.float32,
            initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=False)

        self.saved_mean = tf.Variable(
            name='saved_mean', dtype=tf.float32,
            initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=False)

        self.saved_gamma = tf.Variable(
            name='saved_gamma', dtype=tf.float32,
            initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=False)

        self.saved_beta = tf.Variable(
            name='saved_beta', dtype=tf.float32,
            initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=False)

        self.batch_size_op = tf.shape(self.x)

        # tf.summary.histogram('W', prev_layer_w)
        # tf.summary.histogram('b', prev_layer_b)
        #
        # tf.summary.histogram('gamma', self.gamma)
        # tf.summary.histogram('beta', self.beta)
        #
        # tf.summary.histogram('var', self.var)
        # tf.summary.histogram('mean', self.mean)
        #
        # tf.summary.histogram('saved_var', self.saved_var)
        # tf.summary.histogram('saved_mean', self.saved_mean)
        #
        # tf.summary.histogram('saved_gamma', self.saved_gamma)
        # tf.summary.histogram('saved_beta', self.saved_beta)

        self.save_ops = [tf.assign(self.saved_var, self.var), tf.assign(self.saved_mean, self.mean),
                         tf.assign(self.saved_beta, self.beta), tf.assign(self.saved_gamma, self.gamma)]

        ################# DEFINE transformation ###########################
        # (tf.sqrt(self.var) + self.eps)
        self.bn_to_nobn = [tf.assign(prev_layer_w, (self.saved_gamma / (tf.sqrt(self.saved_var) + self.eps)) * prev_layer_w),
                           tf.assign(prev_layer_b,
                                     (self.saved_gamma / (tf.sqrt(self.saved_var) + self.eps)) * (prev_layer_b - self.saved_mean) + self.saved_beta)]


        # W = W, b = b, gamma=sigma, mean=mu
        self.nobn_to_bn1 = [tf.assign(prev_layer_w, (tf.sqrt(self.saved_var)/ (self.saved_gamma + self.eps)) * prev_layer_w),
                           tf.assign(prev_layer_b,
                                     (tf.sqrt(self.saved_var) / (self.saved_gamma + self.eps)) * (prev_layer_b - self.saved_beta) + self.saved_mean)]

        self.nobn_to_bn2 = [tf.assign(self.gamma, (tf.sqrt(self.var)*self.saved_gamma)/(tf.sqrt(self.saved_var) + self.eps)),
                           tf.assign(self.beta, self.saved_beta + (self.saved_gamma/(tf.sqrt(self.saved_var) + self.eps))*(self.mean - self.saved_mean))]


    #we have 2 main options:
    #1. Do it by the avarage mean/std.
    #2. Do it by the batch mean/std
    def pop_bn(self, sess, fds):
        #sess.run(self.set_is_not_updating[0])
        #
        # _var, _mean, _gamma, _beta = np.zeros([self.x.get_shape()[-1]], np.float32), np.zeros([self.x.get_shape()[-1]], np.float32), \
        #     np.zeros([self.x.get_shape()[-1]], np.float32), np.zeros([self.x.get_shape()[-1]], np.float32)
        # for fd in fds:
        #     _var += sess.run(self.var, fd)
        #     _mean += sess.run(self.mean, fd)/len(fds)
        #     _gamma += sess.run(self.gamma, fd)/len(fds)
        #     _beta += sess.run(self.beta, fd)/len(fds)
        #
        # batch_size = sess.run(self.batch_size_op)[0]
        # _var *= ((batch_size - 1.0) / (batch_size * len(fds) - 1.0))
        # save_fd = {self.var : _var, self.mean : _mean, self.gamma : _gamma, self.beta : _beta}
        # sess.run(self.save_ops, feed_dict=save_fd)
        # sess.run(self.bn_to_nobn, feed_dict=save_fd)

        sess.run(self.save_ops, feed_dict=fds)
        sess.run(self.bn_to_nobn, feed_dict=fds)

        #
        # sess.run(self.save_ops, feed_dict=fd)
        #
        # sess.run(self.bn_to_nobn, feed_dict=fd)
        sess.run(self.set_is_bn_on[0])

        #sess.run(self.set_is_not_updating[1])


    def push_bn(self, sess, fds):
        #sess.run(self.set_is_not_updating[0])

        # _var, _mean = np.zeros([self.x.get_shape()[-1]], np.float32), np.zeros([self.x.get_shape()[-1]], np.float32)
        #
        # for fd in fds:
        #     _mean += sess.run(self.mean, fd)/len(fds)
        #
        # for fd in fds:
        #     _var += sess.run(self.var, fd)
        #
        # batch_size = sess.run(self.batch_size_op)[0]
        # _var *= ((batch_size - 1.0) / (batch_size*len(fds) - 1.0))
        #
        # #print '_var = ' + str(_var)
        # #print '_mean = ' + str(_mean)
        #
        # _save_fd = {self.var : _var, self.mean : _mean}


        sess.run(self.nobn_to_bn1, fds)
        sess.run(self.nobn_to_bn2, fds)

        # sess.run(self.nobn_to_bn1, feed_dict=fd)
        # sess.run(self.nobn_to_bn2, feed_dict=fd)

        #sess.run(self.set_is_not_updating[1])

        sess.run(self.set_is_bn_on[1])
