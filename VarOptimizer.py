
import tensorflow as tf
from my_external_optimizer import ScipyOptimizerInterface
#Parallel block coordinates second order optimizer
class VarOptimizer:
    def __init__(self, model, _vars, loss):
        self.model = model
        self.vars = _vars
        self.loss = loss
        #self.grads = tf.gradients(loss, _vars)

        self.optimizer = ScipyOptimizerInterface(loss=loss, var_list=_vars, \
                                           method='BFGS', options={'maxiter': 10000, 'gtol': 1e-6})

    def run_iteration(self, sess, additional_feed_dict):
        feed_dicts = [self.model.get_shared_feed(sess, [])]
        self.optimizer.minimize(session=sess, feed_dicts=feed_dicts, additional_feed_dict=additional_feed_dict)