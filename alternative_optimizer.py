
import tensorflow as tf
import numpy as np


from tf_utils import avarge_on_feed_dicts

from tf_utils import lazy_property

class AlternativeOptimizer:
    SUMMARY_BEFORE_ITERATION_KEY = 'AlternativeOptimizerDebugBeforeIter'
    SUMMARY_AFTER_ITERATION_KEY = 'AlternativeOptimizerDebugAfterIter'
    SUMMARY_IN_ITERATION_KEY = 'AlternativeOptimizerDebugInIter'


    def var_out(self, var):
        #alpha is the lr for var

        #grad = self.grad_accum.average(self.grad[var])
        out = var - self.grad[var] * self.alphas[var]

        return out


    def build_subspace_graph(self, loss, predictions):

        replacement_ts = {var._snapshot : self.var_out(var) for var in self.orig_var_list}

        if predictions is not None:
            return tf.contrib.graph_editor.graph_replace([loss, predictions], replacement_ts)
        else:
            return tf.contrib.graph_editor.graph_replace([loss], replacement_ts)


    def __init__(self, loss, var_list, **optimizer_kwargs):
        self.eps = 1e-6
        #
        #self.alphas = {var: tf.Variable(var.initialized_value(), dtype=var.dtype.base_dtype, trainable=False) for var in var_list}
        self.alphas = {var : tf.Variable(np.ones(var.get_shape()), dtype=var.dtype.base_dtype, trainable=False) for var in var_list}
        #self.grad_accum = {var: tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype, trainable=False) for var in var_list}






        self.orig_var_list = var_list

        self.lr_for_main = tf.Variable(optimizer_kwargs['lr_for_main'], dtype=tf.float32, trainable=False)
        with tf.name_scope('main_optimizer'):
            if optimizer_kwargs['main_base_method'] == 'SGD':
                self.main_optim = tf.train.GradientDescentOptimizer(self.lr_for_main)
            elif optimizer_kwargs['main_base_method'] == 'Adam':
                self.main_optim = tf.train.AdamOptimizer(self.lr_for_main, optimizer_kwargs['beta1'], optimizer_kwargs['beta2'])
            elif optimizer_kwargs['main_base_method'] == 'Adagrad':
                self.main_optim = tf.train.AdagradOptimizer(self.lr_for_main, initial_accumulator_value=0.001)
            elif optimizer_kwargs['main_base_method'] == 'Adadelta':
                self.main_optim = tf.train.AdadeltaOptimizer(self.lr_for_main, optimizer_kwargs['rho'])
            elif optimizer_kwargs['main_base_method'] == 'Mom':
                self.main_optim = tf.train.MomentumOptimizer(self.lr_for_main, optimizer_kwargs['momentum'])
            else:
                raise Exception('Not implemented!')

            v_grads_and_vars = self.main_optim.compute_gradients(loss, var_list=var_list)
            self.grad = {v: g for g, v in v_grads_and_vars}
            #self.grad_accum = tf.train.ExponentialMovingAverage(decay=0.99)
            #self.grad_maintain_averages_op = self.grad_accum.apply([g for g, v in v_grads_and_vars])
            v_grads_and_vars = [(g * self.alphas[v], v) for g, v in v_grads_and_vars]
            self.main_optim_op = self.main_optim.apply_gradients(v_grads_and_vars)




        if optimizer_kwargs['run_baseline'] == False:

            self.alphas_uniqe = list(set(self.alphas.values()))
            with tf.name_scope('lr_optimization_graph'):
                self.alpha_loss = self.build_subspace_graph(loss, predictions=None)[0]

            #temp = tf.reduce_mean(tf.stack([tf.reduce_mean(self.alphas[v]) for v in self.orig_var_list]))
            #self.lr_for_lr = (self.lr_for_main)/(10.0*temp)
            with tf.name_scope('lr_optimizer'):
                if optimizer_kwargs['lr_base_method'] == 'SGD':
                    self.lr_optim = tf.train.GradientDescentOptimizer(self.lr_for_main)
                elif optimizer_kwargs['lr_base_method'] == 'Adam':
                    self.lr_optim = tf.train.AdamOptimizer(self.lr_for_lr, optimizer_kwargs['beta1'], optimizer_kwargs['beta2'])
                elif optimizer_kwargs['lr_base_method'] == 'Adagrad':
                    self.lr_optim = tf.train.AdagradOptimizer(self.lr_for_lr, initial_accumulator_value=0.001)
                elif optimizer_kwargs['lr_base_method'] == 'Adadelta':
                    self.lr_optim = tf.train.AdadeltaOptimizer(self.lr_for_lr, optimizer_kwargs['rho'])
                elif optimizer_kwargs['lr_base_method'] == 'Mom':
                    self.lr_optim = tf.train.MomentumOptimizer(self.lr_for_lr, optimizer_kwargs['momentum'])
                else:
                    raise Exception('Not implemented!')


                alpha_2_var = {a : v for v,a in self.alphas.items()}
                a_grads_and_vars = self.lr_optim.compute_gradients(self.alpha_loss, var_list=self.alphas_uniqe)
                #self.alpha_accum = tf.train.ExponentialMovingAverage(decay=0.99)
                #self.alpha_maintain_averages_op = self.alpha_accum.apply([v for g, v in a_grads_and_vars])

                with tf.control_dependencies([]):
                #with tf.control_dependencies([self.grad_maintain_averages_op, self.alpha_maintain_averages_op]):

                    # a_grads_and_vars = [(g * self.lr_for_main * v/100.0, v) for g, v in a_grads_and_vars]
                    #grads_and_vars = [(g * alpha_2_var[a], a) for g, a in grads_and_vars]
                    self.lr_optim_op = self.lr_optim.apply_gradients(a_grads_and_vars)


            self.optim_op = [self.lr_optim_op, self.main_optim_op]

        else:
            self.optim_op = [self.main_optim_op]

        tf.summary.scalar('loss', loss, [AlternativeOptimizer.SUMMARY_AFTER_ITERATION_KEY])
        tf.summary.scalar('alpha_loss', self.alpha_loss, [AlternativeOptimizer.SUMMARY_AFTER_ITERATION_KEY])
        self.summary_after = tf.summary.merge_all(AlternativeOptimizer.SUMMARY_AFTER_ITERATION_KEY)
        self.iter = 0


    def set_summary_writer(self, writer):
        self.writer = writer

    #return _sgd_distance_moved, _distance_sesop_moved, _distance_seboost_moved
    def minimize(self, session=None):

        #if self.iter % 180 != 0:
        session.run([self.main_optim_op, self.lr_optim_op])
        #for i in range(180):
        #session.run(self.lr_optim_op)

        # else:
        #     for i in range(180):
        #         session.run(self.lr_optim_op)
        #     session.run(self.main_optim_op)

        self.writer.add_summary(session.run(self.summary_after), self.iter)
        self.iter += 1


