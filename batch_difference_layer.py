
import tensorflow as tf
import numpy as np

class BatchDifferenceLayer:

    def __init__(self, input, labels, n_in, n_out, n_labels, is_training, activation=tf.nn.relu):
        assert (n_out % n_labels == 0)

        #n_labels = 1

        low = -np.sqrt(6.0 / (n_in + n_out / n_labels))  # use 4 for sigmoid, 1 for tanh activation
        high = np.sqrt(6.0 / (n_in + n_out / n_labels))

        # print 'prefix = ' + str(prefix)
        W = tf.Variable(tf.random_uniform([n_in, n_out / n_labels], minval=low, maxval=high), dtype=tf.float32)
        b = tf.Variable(tf.zeros([n_out / n_labels]) + 0.01)
        a = tf.matmul(input, W)  + b
        self.W = W

        outs = []
        self.ema_apply_ops = []

        #self.out = activation(a)
        #return
        for i in range(n_labels):

            #def f(k=i):
            comparison = tf.equal(labels, i)
            #comparison = tf.Print(input_=comparison, data=[comparison], message='comparison_' + str(i) + ' = ', summarize=10)
            out_of_label = tf.where(comparison, a, tf.zeros_like(a))
            params_shape = [out_of_label.get_shape()[-1]]
            #out_of_label = tf.Print(input_=out_of_label, data=[out_of_label], message='out_of_label_' + str(i) + ' = ', summarize=10)

            bn_batch_mean, bn_batch_variance = tf.nn.moments(out_of_label, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update(bn_batch_mean=bn_batch_mean, bn_batch_variance=bn_batch_variance):
                ema_apply_op = ema.apply([bn_batch_mean, bn_batch_variance])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(bn_batch_mean), tf.identity(bn_batch_variance)

            # for testing time
            def moving_mean_var(bn_batch_mean=bn_batch_mean, bn_batch_variance=bn_batch_variance):
                res = (ema.average(bn_batch_mean), ema.average(bn_batch_variance))
                return res

            mean, var = tf.cond(is_training, mean_var_with_update, moving_mean_var)

            beta = tf.Variable(
                name='beta', dtype=tf.float32,
                initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=True)
            #
            # gamma = tf.Variable(
            #     name='gamma', dtype=tf.float32,
            #     initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=True)

            #bn = tf.nn.batch_normalization(a, mean, var, beta, gamma, 0.0001)
            bn = tf.nn.batch_normalization(a, mean, 1.0, beta, None, 0.0000)
            #outs.append(out_of_label)
            #tf.identity(self.bn_batch_mean)
            outs.append(bn)
            #outs.append(a)
            #outs.append( (a - mean)/ (tf.sqrt(var) + 0.0001))

        out = tf.stack(outs, axis=1)
        self.out = tf.reshape(out, [-1, n_out])
        self.out = activation(self.out)

        #self.out = tf.Print(input_=self.out, data=[self.out], message='self.out' + ' = ', summarize=10)

