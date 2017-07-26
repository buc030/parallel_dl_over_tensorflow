
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

    def set_summary_writer(self, writer):
        self.writer = writer

    def dot(self, xs, ys):
        return tf.add_n([tf.reduce_sum(x*y) for x,y in zip(xs, ys)])

    @lazy_property
    def lr_update_multiplier(self):
        if self.per_variable == False:
            return self.dot([var - self.snapshot_curr[var] for var in self.var_list], [self.snapshot_curr[var] - self.snapshot_prev[var] for var in self.var_list]) \
                / (tf.global_norm([var - self.snapshot_curr[var] for var in self.var_list])*tf.global_norm([self.snapshot_curr[var] - self.snapshot_prev[var]  for var in self.var_list]))

        else:
            return tf.stack([self.dot([var - self.snapshot_curr[var]], [self.snapshot_curr[var] - self.snapshot_prev[var]])/
                             (tf.norm(var - self.snapshot_curr[var])*tf.norm(self.snapshot_curr[var] - self.snapshot_prev[var])) for var in self.var_list])


    @lazy_property
    def shift_snapshots(self):
        return [tf.assign(self.snapshot_prev[var], self.snapshot_curr[var]) for var in self.var_list]

    @lazy_property
    def take_snapshot_curr(self):
        return [tf.assign(self.snapshot_curr[var], var) for var in self.var_list]

    def __init__(self, loss, batch_provider, var_list, **optimizer_kwargs):

        self.input = batch_provider.batch()
        self.bp = batch_provider
        self.loss = loss
        self.best_loss = 10000.0
        self.var_list = var_list


        self.batch_size = optimizer_kwargs['batch_size']
        self.train_dataset_size = optimizer_kwargs['train_dataset_size']

        self.iters_per_adjust = optimizer_kwargs['iters_per_adjust']
        self.iters_to_wait_before_first_collect = optimizer_kwargs['iters_to_wait_before_first_collect']
        self.per_variable = optimizer_kwargs['per_variable']
        self.base_optimizer = optimizer_kwargs['base_optimizer']

        initial_lr = optimizer_kwargs['lr']
        if self.per_variable == False:
            self.lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
            self.curr_lr = initial_lr
            self.prev_lr = initial_lr
            self.lr_placeholder = tf.placeholder(dtype=tf.float32)
            self.update_lr = tf.assign(self.lr, self.lr_placeholder)
        else:
            self.lr = tf.Variable(np.ones(len(var_list))*initial_lr, dtype=tf.float32, trainable=False)
            self.curr_lr = initial_lr*np.ones(len(var_list))
            self.prev_lr = initial_lr * np.ones(len(var_list))
            self.lr_placeholder = tf.placeholder(dtype=tf.float32)
            self.update_lr = tf.assign(self.lr, self.lr_placeholder)


        with tf.name_scope('snapshot_curr'):
            self.snapshot_curr = {var : tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype) for var in var_list }

        with tf.name_scope('snapshot_prev'):
            self.snapshot_prev = {var : tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype) for var in var_list }

        self.lr_update_multiplier

        with tf.name_scope('base_optimizer'):
            if self.base_optimizer == 'SGD':
                self.sgd_optim = tf.train.GradientDescentOptimizer(self.lr)
            elif self.base_optimizer == 'Adam':
                self.sgd_optim = tf.train.AdamOptimizer(self.lr, optimizer_kwargs['beta1'], optimizer_kwargs['beta2'])
            elif self.base_optimizer == 'Adagrad':
                self.sgd_optim = tf.train.AdagradOptimizer(self.lr)
            elif self.base_optimizer == 'Adadelta':
                self.sgd_optim = tf.train.AdadeltaOptimizer(self.lr, optimizer_kwargs['rho'])
            else:
                raise Exception('Not implemented!')

            self.sgd_grads = tf.gradients(loss, var_list)
            self.sgd_op = self.sgd_optim.minimize(loss)
            # if self.per_variable == False:
            #     multiplied_grads = [self.lr*self.sgd_grads[i] for i in range(len(self.sgd_grads))]
            # else:
            #     multiplied_grads = [self.lr[i]*self.sgd_grads[i] for i in range(len(self.sgd_grads))]
            # self.sgd_op = self.sgd_optim.apply_gradients(zip(multiplied_grads, var_list), name='train_step')


            tf.summary.scalar('weights_norm_during_sgd', tf.global_norm(var_list), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('grad_norm_norm_during_sgd', tf.global_norm(self.sgd_grads), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('grad_norm_norm_during_sgd_normalized_by_batch_size', tf.global_norm(self.sgd_grads)*tf.sqrt(float(self.batch_size)), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            if self.per_variable == False:
                tf.summary.scalar('learning_rate', self.lr, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
            else:
                for i in range(len(var_list)):
                    tf.summary.scalar('learning_rate_' + var_list[i].name, self.lr[i], [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

            if self.per_variable == True:
                for i in range(len(var_list)):
                    tf.summary.scalar('lr_update_multiplier_' + var_list[i].name, self.lr_update_multiplier[i], [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

                    tf.summary.scalar('norm_' + var_list[i].name, tf.norm(self.snapshot_curr[var_list[i]] - self.snapshot_prev[var_list[i]])**2, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
                    tf.summary.scalar('dot_' + var_list[i].name,
                                      self.dot([var_list[i] - self.snapshot_curr[var_list[i]]], [self.snapshot_curr[var_list[i]] - self.snapshot_prev[var_list[i]]]),
                                      [SgdAdjustOptimizer.SUMMARY_SGD_KEY])


        self.sgd_summaries = tf.summary.merge_all(SgdAdjustOptimizer.SUMMARY_SGD_KEY)

        self.explotion_counter = 0
        self.iter = 0
        self.iter_runner = None

    def _run_iter(self, sess):


        print 'self.iter = ' + str(self.iter)
        if self.iter < self.iters_to_wait_before_first_collect:
            self.writer.add_summary(sess.run(self.sgd_summaries), self.iter)
            sess.run(self.sgd_op)
            self.iter += 1
            return
        #Early stopping in case of divegence
        # if curr_loss > 100*self.best_loss or np.isnan(curr_loss):
        #     self.explotion_counter += 1
        #     if self.explotion_counter > 5:
        #         raise Exception('Optimization has exploded, stoping early!')
        # else:
        #     self.explotion_counter = 0

        sess.run(self.take_snapshot_curr)
        for i in range(int(self.iters_per_adjust) - 1):
            self.writer.add_summary(sess.run(self.sgd_summaries), self.iters_to_wait_before_first_collect + (self.iter - self.iters_to_wait_before_first_collect)* int(self.iters_per_adjust) + i)
            sess.run(self.sgd_op)
            yield


        self.writer.add_summary(sess.run(self.sgd_summaries), self.iters_to_wait_before_first_collect + (self.iter - self.iters_to_wait_before_first_collect) * int(self.iters_per_adjust) + self.iters_per_adjust)
        sess.run(self.sgd_op)

        if self.iter == self.iters_to_wait_before_first_collect:
            sess.run(self.shift_snapshots)
            self.iter += 1
            return

        _lr_update_multiplier = sess.run(self.lr_update_multiplier)

        print '_lr_update_multiplier = ' + str(_lr_update_multiplier)
        #at the begining, update using
        #next_lrs = (self.curr_lr/self.prev_lr) + self.prev_lr * _lr_update_multiplier

        next_lrs = self.curr_lr * (1.0 +  _lr_update_multiplier)


        #SV DEBUG
        # next_lrs1 = self.curr_lr * _lr_update_multiplier + self.prev_lr
        # next_lrs2 = self.prev_lr * _lr_update_multiplier  + self.curr_lr
        # next_lrs = min(next_lrs1, next_lrs2)

        if self.per_variable == True:
            negetive_idx = np.where(next_lrs <= 0.0)
            next_lrs[negetive_idx] = self.curr_lr[negetive_idx]
        else:
            assert (next_lrs >= 0)
            if next_lrs == 0:
                next_lrs = self.curr_lr

        self.prev_lr = self.curr_lr
        self.curr_lr = next_lrs
        print 'setting lr to: ' + str(self.curr_lr)
        sess.run(self.update_lr, {self.lr_placeholder: self.curr_lr})


        sess.run(self.shift_snapshots)
        print '------------------'

        self.iter += 1

    def run_iter(self, sess):
        if self.iter_runner is None:
            self.iter_runner = self._run_iter(sess)
        try:
            self.iter_runner.next()
        except StopIteration:
            self.iter_runner = None


