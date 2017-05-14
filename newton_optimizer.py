
import tensorflow as tf
import tf_utils
from my_external_optimizer import ScipyOptimizerInterface
import numpy as np

def _get_shape_tuple(tensor):
  return tuple(dim.value for dim in tensor.get_shape())

class NewtonOptimizer:

    # y(wrts):R^n -> R^m
    # so the shape of the jacobian is m x n
    def calc_jacobian(self, y, wrts):

        for wrt in wrts:
            assert wrt.get_shape() == 1
        jacobian = []
        for logit in y:
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
    def __init__(self, loss, train_variables):
        assert (False) #Not implemented at all. Ignore the code.
        self.loss = loss
        self.unstacked_train_variables = train_variables
        self.train_variables = tf.stack(train_variables)

        self.train_variables = tf.reshape(self.train_variables, [len(train_variables)])

        self.grad = tf.gradients(loss, train_variables)
        self.hessian = self.calc_jacobian(self.grad, train_variables)
        #self.hessian = self.hessian + tf.diag(tf.ones([self.hessian.get_shape()[0]])*1e-16)
        self.hessian_norm = tf.norm(self.hessian)
        self.hessian = tf.Print(self.hessian, [self.hessian])

        self.gradient_norm = tf.global_norm(self.grad)

        self.c = tf.Variable(initial_value=np.ones(1), dtype=tf.float32)
        self.c_loss = tf.square((self.gradient_norm / (self.hessian_norm + 1e-16)) - self.c)

        self.cg_on_c_loss = ScipyOptimizerInterface(loss=self.c_loss, var_list=[self.c], \
                                          method='BFGS', options={'maxiter': 200, 'gtol': 1e-6})


        self.cg = ScipyOptimizerInterface(loss=loss, var_list=train_variables, \
                                          method='BFGS', options={'maxiter': 200 * len(train_variables), 'gtol': 1e-6})

        self.cg.c = self.c

    #eval h(x) = f(cx) = ||c - grad(cx)/hessian(cx)||
    def eval_func(self, x):
        """Function to evaluate a `Tensor`."""

        tensors = [self.cg_on_c_loss._loss, self.cg_on_c_loss._packed_loss_grad]
        fetches = []
        augmented_fetches = tensors + fetches

        c = self.sess.run(self.c)
        # x[self._packing_slices[0]].reshape(_get_shape_tuple(self._vars[0]))*800000
        augmented_feed_dict = {
                var: x[packing_slice].reshape(_get_shape_tuple(var))
                for var, packing_slice in zip(self.cg_on_c_loss._vars, self.cg_on_c_loss._packing_slices)
            }

        for alpha in self.unstacked_train_variables:
            augmented_feed_dict[alpha] = self.sess.run(alpha)*c

        res = tf_utils.avarge_on_feed_dicts(sess=self.sess, target_ops=augmented_fetches,
                                       feed_dicts=self.feed_dicts, additional_feed_dict=augmented_feed_dict)


        assert (len(res) == 2)
        return res

    def minimize(self, sess, feed_dicts, loss_callback=None):

        self.sess = sess
        self.feed_dicts = feed_dicts

        self.cg_on_c_loss.c = 1.0

        print 'c before opimization: ' + str(sess.run(self.c))
        self.cg_on_c_loss.minimize(session=sess, feed_dicts=feed_dicts,
                         loss_callback=loss_callback, override_loss_grad_func=self.eval_func)
        print 'c after opimization: ' + str(sess.run(self.c))
        self.cg.c = sess.run(self.c)

        print 'grad norm = ' + str(tf_utils.avarge_on_feed_dicts(sess=self.sess, target_ops=[self.gradient_norm],
                                       feed_dicts=self.feed_dicts, additional_feed_dict={alpha : self.sess.run(alpha)*sess.run(self.c)
                                                                                         for alpha in self.unstacked_train_variables}))

        print 'grad norm = ' + str(tf_utils.avarge_on_feed_dicts(sess=self.sess, target_ops=[self.gradient_norm],
                                                                 feed_dicts=self.feed_dicts, additional_feed_dict={
            alpha: [8]
            for alpha in self.unstacked_train_variables}))

        print 'grad norm = ' + str(tf_utils.avarge_on_feed_dicts(sess=self.sess, target_ops=[self.gradient_norm],
                                                                 feed_dicts=self.feed_dicts))

        print 'grad norm = ' + str(tf_utils.avarge_on_feed_dicts(sess=self.sess, target_ops=[self.gradient_norm],
                                                                 feed_dicts=self.feed_dicts, additional_feed_dict={
            alpha: self.sess.run(alpha)
            for alpha in self.unstacked_train_variables}))


        return self.cg.minimize(session=sess, feed_dicts=feed_dicts,
               loss_callback=loss_callback)