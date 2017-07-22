
import tensorflow as tf
from tensorflow.python.ops import array_ops
from my_external_optimizer import ScipyOptimizerInterface

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
        assert (len(wrts) == 1)
        return tf.stack([tf.gradients(y[i], wrts)[0] for i in range(y.get_shape()[1])])

        # for wrt in wrts:
        #     assert wrt.get_shape() == 1
        # jacobian = []
        # for logit in array_ops.unstack(y, axis=1):
        #     # print 'logit = ' + str(logit)
        #     logit_grad = tf.stack(tf.gradients(logit, wrts))
        #
        #     # Make sure gradient is a column vector!
        #     assert (logit_grad.get_shape()[1] == 1)
        #     logit_grad = tf.reshape(logit_grad, [tf.shape(logit_grad)[0]])
        #     # print 'y_i gradient = ' + str(logit_grad)
        #     jacobian.append(logit_grad / tf.cast(tf.shape(logit)[0], tf.float32))
        #
        # # y is a vector of size 4, so dydW[i] is the gradient of y_i
        # return tf.stack(jacobian)

    def get_packed_loss_grad(self):
        return self.cg._packed_loss_grad

    #logit has to be softmax!
    #Every train var must be a scalar!
    def __init__(self, loss, logits, train_variables, options):
        self.loss = loss
        self.logits = logits
        self.train_variables = train_variables
        assert (len(train_variables) == 1)
        assert (len(train_variables[0].get_shape()) == 1)
        self.jacobian = self.calc_jacobian(logits, train_variables)

        temp = tf.matmul(tf.transpose(self.jacobian), tf.diag(1.0 / tf.reduce_mean(logits, axis=0)))
        self.F = tf.matmul(temp, self.jacobian)

        self.v = tf.placeholder(tf.float32, shape=[train_variables[0].get_shape()[0]])
        v = tf.reshape(self.v, shape=[int(train_variables[0].get_shape()[0]), 1])
        self.Fv = tf.matmul(self.F, v)

        self.cg = ScipyOptimizerInterface(loss=self.loss, var_list=train_variables,
            hessp=self.fisher_information_matrix, \
            method='trust-ncg', options=options)

    def minimize(self, session, feed_dicts, fetches=None,
               step_callback=None, loss_callback=None, override_loss_grad_func=None, additional_feed_dict=None):
        if additional_feed_dict is None:
            additional_feed_dict = {}
        self.sess = session
        self.cg.minimize(self.sess, feed_dicts=feed_dicts, additional_feed_dict=additional_feed_dict, loss_callback=loss_callback)