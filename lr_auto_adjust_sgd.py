
import tensorflow as tf
import numpy as np

from tf_utils import avarge_on_feed_dicts, avarge_n_calls, lazy_property

# Usage:
# ----------
# run_epoch
# run_sesop

# run_epoch
# run_sesop
class SgdAdjustOptimizer:
    SUMMARY_SGD_KEY = 'sgd_debug_summaries'
    SUMMARY_AFTER_SGD_KEY = 'after_sgd_debug_summaries'

    def set_summary_writer(self, writer):
        self.writer = writer

    def dot(self, xs, ys):
        return tf.add_n([tf.reduce_sum(x*y) for x,y in zip(xs, ys)])

    def mult_list(self, xs, ys):
        return [x * y for x, y in zip(xs, ys)]

    def add_list(self, xs, ys):
        return [x + y for x, y in zip(xs, ys)]

    def assign_list(self, xs, ys):
        return [tf.assign(x, y) for x, y in zip(xs, ys)]

    def maximum_list(self, xs, y):
        return [tf.maximum(x, y) for x in xs]

    def where_positive_list(self, masks, x, y):
        return [tf.where(m > 0, tf.ones_like(m)*x , tf.ones_like(m)*y) for m in masks]


    @lazy_property
    def v(self):
        return [var - self.snapshot_curr[var] for var in self.var_list]

    @lazy_property
    def u(self):
        return [self.snapshot_curr[var] - self.snapshot_prev[var] for var in self.var_list]


    @lazy_property
    def lr_update_multiplier(self):
        if self.per_variable == False:
            return self.dot(self.v, self.u) / (tf.global_norm(self.v)*tf.global_norm(self.u))
        else:
            return [tf.sign((var - self.snapshot_curr[var])*(self.snapshot_curr[var] - self.snapshot_prev[var])) for var in self.var_list]

            # return tf.stack([((var - self.snapshot_curr[var])*(self.snapshot_curr[var] - self.snapshot_prev[var]))/
            #                   (tf.abs(var - self.snapshot_curr[var])*tf.abs(self.snapshot_curr[var] - self.snapshot_prev[var])) for var in self.var_list])

            # return tf.stack([self.dot([var - self.snapshot_curr[var]], [self.snapshot_curr[var] - self.snapshot_prev[var]])/
            #                  (tf.norm(var - self.snapshot_curr[var])*tf.norm(self.snapshot_curr[var] - self.snapshot_prev[var])) for var in self.var_list])


    @lazy_property
    def shift_snapshots(self):
        return [tf.assign(self.snapshot_prev[var], self.snapshot_curr[var]) for var in self.var_list]

    @lazy_property
    def take_snapshot_curr(self):
        return [tf.assign(self.snapshot_curr[var], var) for var in self.var_list]

    def __init__(self, loss, batch_provider, var_list, **optimizer_kwargs):

        # self.input = batch_provider.batch()
        # self.bp = batch_provider
        self.loss = loss
        # self.best_loss = 10000.0
        self.var_list = var_list

        self.optimizer_kwargs = optimizer_kwargs
        #self.batch_size = optimizer_kwargs['batch_size']
        #self.train_dataset_size = optimizer_kwargs['train_dataset_size']

        self.iters_per_adjust = optimizer_kwargs['iters_per_adjust']
        self.iters_to_wait_before_first_collect = optimizer_kwargs['iters_to_wait_before_first_collect']
        #self.iters_to_wait_before_first_collect = max(1, self.iters_to_wait_before_first_collect)
        self.per_variable = optimizer_kwargs['per_variable']
        self.base_optimizer = optimizer_kwargs['base_optimizer']

        with tf.name_scope('snapshot_curr'):
            self.snapshot_curr = {
            var: tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype, trainable=False) for var in
            var_list}

        with tf.name_scope('snapshot_prev'):
            self.snapshot_prev = {
            var: tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype, trainable=False) for var in
            var_list}

        initial_lr = optimizer_kwargs['lr']
        self.sgd_counter = tf.Variable(0, dtype=tf.float32, trainable=False)
        if self.per_variable == False:
            self.lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
            self.lr_placeholder = tf.placeholder(dtype=tf.float32)
            self.update_lr = tf.assign(self.lr, self.lr_placeholder)
            if optimizer_kwargs['step_size_anealing'] == True:
                self.lr = self.lr / tf.sqrt(self.sgd_counter + 1)

        else:

            self.lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
            self.per_var_lr = [tf.Variable(tf.ones_like(var), dtype=var.dtype.base_dtype, trainable=False) for var in var_list]
            self.prev_per_var_lr = [tf.Variable(tf.ones_like(var), dtype=var.dtype.base_dtype, trainable=False) for var in var_list]

            #next_lr = self.per_var_lr*self.lr_update_multiplier + self.prev_per_var_lr

            next_lr = self.mult_list(self.where_positive_list(self.lr_update_multiplier, 1.1, 0.5), self.per_var_lr)

            #next_lr = self.mult_list(self.per_var_lr, should_go_faster)

            #next_lr = self.maximum_list(self.add_list(self.mult_list(self.per_var_lr, self.lr_update_multiplier), self.prev_per_var_lr), 1.0)

            #next_lr = self.maximum_list(self.add_list(self.mult_list(self.prev_per_var_lr, self.lr_update_multiplier), self.per_var_lr), 1e-6)

            with tf.control_dependencies(next_lr):
                with tf.control_dependencies(self.assign_list(self.prev_per_var_lr, self.per_var_lr)):
                    self.update_lr = self.assign_list(self.per_var_lr, next_lr)


        with tf.name_scope('base_optimizer'):
            if self.base_optimizer == 'SGD':
                self.sgd_optim = tf.train.GradientDescentOptimizer(self.lr)
            elif self.base_optimizer == 'Adam':
                self.sgd_optim = tf.train.AdamOptimizer(self.lr, optimizer_kwargs['beta1'], optimizer_kwargs['beta2'])
            elif self.base_optimizer == 'Momentum':
                self.sgd_optim = tf.train.MomentumOptimizer(self.lr, optimizer_kwargs['momentum'], use_nesterov=False)
            elif self.base_optimizer == 'Nestrov':
                self.sgd_optim = tf.train.MomentumOptimizer(self.lr, optimizer_kwargs['momentum'], use_nesterov=True)
            elif self.base_optimizer == 'Adagrad':
                self.sgd_optim = tf.train.AdagradOptimizer(self.lr)
            elif self.base_optimizer == 'Adadelta':
                self.sgd_optim = tf.train.AdadeltaOptimizer(self.lr, optimizer_kwargs['rho'])
            else:
                raise Exception('Not implemented!')


            # accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in var_list]
            # zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            # gvs = self.sgd_optim.compute_gradients(rmse, tvs)
            # accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
            # train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])



            # if self.per_variable == False:
            #     multiplied_grads = [self.lr*self.sgd_grads[i] for i in range(len(self.sgd_grads))]
            # else:
            #     multiplied_grads = [self.lr[i]*self.sgd_grads[i] for i in range(len(self.sgd_grads))]
            # self.sgd_op = self.sgd_optim.apply_gradients(zip(multiplied_grads, var_list), name='train_step')


            self.weight_norm_snapshot = tf.Variable(0.0, dtype=tf.float32, trainable=False)

            self.take_weight_norm_snapshot = tf.assign(self.weight_norm_snapshot, tf.global_norm(var_list))

            self.weight_norm_slope = (tf.global_norm(var_list) - self.weight_norm_snapshot)/(self.sgd_counter)


            if 'grads_vars' in optimizer_kwargs:
                grads_vars = optimizer_kwargs['grads_vars']
            else:
                grads_vars = self.sgd_optim.compute_gradients(loss, var_list)



            if 'normalize_gradients' in optimizer_kwargs and optimizer_kwargs['normalize_gradients'] == True:
                norm = tf.global_norm([g for (g, v) in grads_vars])
                grads_vars = [(g/norm, v) for (g, v) in grads_vars]

            self.sgd_grads = [g for (g, v) in grads_vars]

            if self.per_variable == True:
                multiplied_grads = self.mult_list(self.sgd_grads, self.per_var_lr)
                self.sgd_op = self.sgd_optim.apply_gradients([(g, v) for g,v in zip(multiplied_grads, var_list)], global_step=self.sgd_counter)
            else:
                self.sgd_op = self.sgd_optim.apply_gradients(grads_vars, global_step=self.sgd_counter)

            # self.weight_norm_slope_update = \
            #         tf.assign(self.weight_norm_slope,
            #                   ((tf.global_norm(self.var_list) - self.weight_norm_snapshot) / (self.sgd_counter - 2*self.iters_per_adjust - self.iters_to_wait_before_first_collect + 1)))

            self.train_ops = [self.sgd_op]

            tf.summary.scalar('sgd_counter', self.sgd_counter, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            tf.summary.scalar('weight_norm_slope', self.weight_norm_slope, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('weights_norm_during_sgd', tf.global_norm(var_list), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('grad_norm_norm_during_sgd', tf.global_norm(self.sgd_grads), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            tf.summary.scalar('v norm', tf.global_norm(self.v), [SgdAdjustOptimizer.SUMMARY_AFTER_SGD_KEY])
            tf.summary.scalar('u norm', tf.global_norm(self.u), [SgdAdjustOptimizer.SUMMARY_AFTER_SGD_KEY])
            tf.summary.scalar('dot (u, v)', self.dot(self.v, self.u), [SgdAdjustOptimizer.SUMMARY_AFTER_SGD_KEY])

            tf.summary.scalar('angle (u, v)',
                              self.dot(self.v, self.u) \
                              / (tf.global_norm(self.v) * tf.global_norm(self.u)), [SgdAdjustOptimizer.SUMMARY_AFTER_SGD_KEY])



            #tf.summary.scalar('grad_norm_norm_during_sgd_normalized_by_batch_size', tf.global_norm(self.sgd_grads)*tf.sqrt(float(self.batch_size)), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            # for i in range(32):
            #     tf.summary.scalar('weight_flactuations', var_list[0][0][0][0][i], [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            if self.per_variable == False:
                tf.summary.scalar('learning_rate', self.lr, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            # else:
            #     for i in range(len(var_list)):
            #         tf.summary.scalar('learning_rate_' + var_list[i].name, self.lr[i], [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            # if self.per_variable == True:
            #     for i in range(len(var_list)):
            #         tf.summary.scalar('lr_update_multiplier_' + var_list[i].name, self.lr_update_multiplier[i], [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            #
            #         tf.summary.scalar('norm_' + var_list[i].name, tf.norm(self.snapshot_curr[var_list[i]] - self.snapshot_prev[var_list[i]])**2, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            #         tf.summary.scalar('dot_' + var_list[i].name,
            #                           self.dot([var_list[i] - self.snapshot_curr[var_list[i]]], [self.snapshot_curr[var_list[i]] - self.snapshot_prev[var_list[i]]]),
            #                           [SgdAdjustOptimizer.SUMMARY_SGD_KEY])


        self.sgd_summaries = tf.summary.merge_all(SgdAdjustOptimizer.SUMMARY_SGD_KEY)
        self.after_sgd_summaries = tf.summary.merge_all(SgdAdjustOptimizer.SUMMARY_AFTER_SGD_KEY)
        self.after_sgd_summary_idx = 0
        self.summary_idx = 0

        self.explotion_counter = 0
        self.iter = 0
        self.iter_runner = None
        self.transition_iter = 0
        self.sgd_iter = 0
        self.number_of_steps_for_smooth_transition = None

        self.lrs = [initial_lr, initial_lr, initial_lr]
        #lrs[0] is lr_n
        #lrs[1] is lr_(n-1)
        # lrs[2] is lr_(n-2)

        self.lr_update_multiplier
        self.shift_snapshots
        self.take_snapshot_curr

        self._lr_update_multiplier = 0.0

        self.num_of_snapshots_taken = 0




    def get_actual_curr_lr(self):
        if self.number_of_steps_for_smooth_transition is None:
            return self.lrs[0]

        a = self.transition_iter / float(self.number_of_steps_for_smooth_transition)
        return self.lrs[1] * (1.0 - a) + self.lrs[0] * a

    #return true iff lr transit has finished
    def do_lr_transit(self, sess):
        self.transition_iter += 1
        sess.run(self.update_lr, {self.lr_placeholder: self.get_actual_curr_lr() })
        return self.transition_iter == self.number_of_steps_for_smooth_transition


    def _run_sgd_iter(self, sess):
        if self.sgd_iter == 0:
            sess.run(self.take_weight_norm_snapshot)

        if self.number_of_steps_for_smooth_transition is not None and self.per_variable == False:
            if self.do_lr_transit(sess):
                self.number_of_steps_for_smooth_transition = None
                # self.transition_iter = 0

        #take snapshot
        if self.sgd_iter >= self.iters_to_wait_before_first_collect and \
                (self.sgd_iter - self.iters_to_wait_before_first_collect) % int(self.iters_per_adjust) == 0:
            self.writer.add_summary(sess.run(self.after_sgd_summaries), self.after_sgd_summary_idx)
            self.after_sgd_summary_idx += 1
            sess.run(self.shift_snapshots)
            sess.run(self.take_snapshot_curr)
            self.num_of_snapshots_taken += 1


        self.writer.add_summary(sess.run(self.sgd_summaries), self.summary_idx)
        self.summary_idx += 1
        sess.run(self.train_ops, self.feed_dict)

        self.sgd_iter += 1

        if self.optimizer_kwargs['disable_lr_change']:
            return False

        if self.num_of_snapshots_taken < 2:
            return False

        if self.optimizer_kwargs['update_every_two_snapshots'] == True:
            return (self.num_of_snapshots_taken % 2) == 0 and (self.sgd_iter - self.iters_to_wait_before_first_collect) % int(self.iters_per_adjust) == 0

        return (self.sgd_iter - self.iters_to_wait_before_first_collect) % int(self.iters_per_adjust) == 0


    def _run_iter(self, sess):


        print 'self.iter = ' + str(self.iter)

        while not self._run_sgd_iter(sess):
            yield



        if self.per_variable == True:
            sess.run(self.update_lr)
            print 'lr = ' + str(sess.run(self.per_var_lr))

            self.iter += 1
            return

        _lr_update_multiplier = sess.run(self.lr_update_multiplier)

        print '_lr_update_multiplier = ' + str(_lr_update_multiplier)
        if self.optimizer_kwargs['reduce_lr_only'] == True and _lr_update_multiplier > 0:
            _lr_update_multiplier = 0.0
            print 'Not allowed to increase learning rate setting _lr_update_multiplier = ' + str(_lr_update_multiplier)
        #SV DEBUG
        # #decay = 1.0/10.0**(self.sgd_iter/5000)
        # decay = 1.0
        # self._lr_update_multiplier = self._lr_update_multiplier*(1 - decay) + _lr_update_multiplier*decay
        #
        # print '1 - decay = ' + str(1 - decay)

        # print 'accumulated _lr_update_multiplier = ' + str(self._lr_update_multiplier)
        #
        # _lr_update_multiplier -= self._lr_update_multiplier
        # print '_lr_update_multiplier after add = ' + str(_lr_update_multiplier)
        # if abs(_lr_update_multiplier) < 0.5:
        #     self.iter += 1
        #     return

        #at the begining, update using
        #next_lrs = (self.curr_lr/self.prev_lr) + self.prev_lr * _lr_update_multiplier

        #next_lrs = self.get_actual_curr_lr() * (1.0 +  _lr_update_multiplier)
        if self.optimizer_kwargs['lr_update_formula_risky'] == True:
            next_lrs = (1 + _lr_update_multiplier)*self.lrs[0]
            #next_lrs = min(abs(_lr_update_multiplier * self.lrs[0] + self.lrs[1]), abs(1.0/_lr_update_multiplier * self.lrs[0] + self.lrs[1]))
            #opt1 = _lr_update_multiplier * self.lrs[0] + self.lrs[1]
            #opt2 = 1.0/_lr_update_multiplier * self.lrs[0] + self.lrs[1]
            # opt1 = (1.0 + _lr_update_multiplier) * self.lrs[0]
            # opt2 = (1.0 + max(1.0/_lr_update_multiplier, -0.9)) * self.lrs[0]
            # if opt1 < opt2:
            #     print 'chose opt1'
            # else:
            #     print 'chose opt2'
            # next_lrs = min(opt1, opt2)
            #next_lrs = (opt1 + opt2)*0.5

        else:
            next_lrs = (_lr_update_multiplier * (self.lrs[0] + self.lrs[1]) + self.lrs[1] + self.lrs[2]) / 2
        #self.iters_per_adjust = self.number_of_steps_for_smooth_transition


        if self.per_variable == True:
            negetive_idx = np.where(next_lrs <= 0.0)
            next_lrs[negetive_idx] = self.get_actual_curr_lr()[negetive_idx]
        else:
            assert (next_lrs >= 0)
            if next_lrs == 0:
                next_lrs = self.get_actual_curr_lr()

        # if we are in the middle of transistioning, simply continue the transitioning from where we were
        # self.prev_lr = self.get_actual_curr_lr()
        # self.curr_lr = next_lrs

        print 'target lr: ' + str(next_lrs)


        # self.number_of_steps_for_smooth_transition = int(
        #     abs(next_lrs - self.get_actual_curr_lr()) / abs(sess.run(self.weight_norm_slope))) + 1
        #
        # # SV DEBUG
        # self.number_of_steps_for_smooth_transition = max(self.number_of_steps_for_smooth_transition,
        #                                                  self.iters_per_adjust)
        self.number_of_steps_for_smooth_transition = self.iters_per_adjust
        print 'number of steps to smooth tranition: ' + str(self.number_of_steps_for_smooth_transition)

        #self.lrs = [next_lrs] + self.lrs[0:-1]
        self.lrs = [next_lrs, self.get_actual_curr_lr(), self.lrs[1]]
        self.transition_iter = 0
        #sess.run(self.shift_snapshots)
        print '------------------'

        self.iter += 1

    def run_iter(self, sess, feed_dict=None):
        if self.iter_runner is None:
            self.iter_runner = self._run_iter(sess)
        try:
            if feed_dict == None:
                self.feed_dict = {}
            else:
                self.feed_dict = feed_dict
            self.iter_runner.next()
        except StopIteration:
            self.iter_runner = None


