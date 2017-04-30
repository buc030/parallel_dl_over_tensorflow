
import tensorflow as tf
from tensorflow.python.ops import array_ops

def _get_shape_tuple(tensor):
  return tuple(dim.value for dim in tensor.get_shape())

class NaturalGradientOptimizer:

    #compute F*p
    def fisher_information_matrix(self, x, p):
        augmented_feed_dict = {
            var: x[packing_slice].reshape(_get_shape_tuple(var))
            for var, packing_slice in zip(self.cg._vars, self.cg._packing_slices)
        }

        augmented_feed_dict.update({self.v : p})
        augmented_fetch_vals = self.sess.run(self.Fv, feed_dict=augmented_feed_dict)

        #print 'p = ' + str(p)
        #print 'augmented_fetch_vals = ' + str(augmented_fetch_vals)

        return augmented_fetch_vals.flatten()

    # y(wrts):R^n -> R^m
    # so the shape of the jacobian is m x n
    def calc_jacobian(self, y, wrts):

        for wrt in wrts:
            assert wrt.get_shape() == 1
        jacobian = []
        for logit in array_ops.unstack(y, axis=1):
            # print 'logit = ' + str(logit)
            logit_grad = tf.stack(tf.gradients(logit, wrts))

            # Make sure gradient is a column vector!
            assert (logit_grad.get_shape()[1] == 1)
            logit_grad = tf.reshape(logit_grad, [tf.shape(logit_grad)[0]])
            # print 'y_i gradient = ' + str(logit_grad)
            jacobian.append(logit_grad / tf.cast(tf.shape(logit)[0], tf.float32))

        # y is a vector of size 4, so dydW[i] is the gradient of y_i
        return tf.stack(jacobian)

    #logit has to be softmax!
    #Every train var must be a scalar!
    def __init__(self, loss, logits, train_variables):
        self.loss = loss
        self.logits = logits
        self.train_variables = train_variables

        self.jacobian = self.calc_jacobian(logits, train_variables)

        temp = tf.matmul(tf.transpose(self.jacobian), tf.diag(1.0 / tf.reduce_mean(logits, axis=0)))
        self.F = tf.matmul(temp, self.jacobian)

        self.v = tf.placeholder(tf.float32, shape=[len(train_variables)])
        v = tf.reshape(self.v, [len(train_variables), 1])
        self.Fv = tf.matmul(self.F, v)

        self.cg = tf.contrib.opt.ScipyOptimizerInterface(loss=self.loss, var_list=train_variables,
            iteration_mult=None, hessp=self.fisher_information_matrix, \
            method='trust-ncg', options={'maxiter': 10})

    def minimize(self, sess, feed_dicts, loss_callback=None):
        self.sess = sess
        self.cg.minimize(self.sess, feed_dicts=feed_dicts)