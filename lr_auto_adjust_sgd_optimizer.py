
import tensorflow as tf
import numpy as np

from tf_utils import avarge_on_feed_dicts, avarge_n_calls, lazy_property
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
# Usage:
# ----------
# run_epoch
# run_sesop

# run_epoch
# run_sesop
class SgdAdjustOptimizer(optimizer.Optimizer):
    SUMMARY_SGD_KEY = 'sgd_debug_summaries'
    SUMMARY_AFTER_SGD_KEY = 'after_sgd_debug_summaries'

    def set_summary_writer(self, writer):
        self.writer = writer

    def __init__(self, var_list, **optimizer_kwargs):
    #def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(SgdAdjustOptimizer, self).__init__(optimizer_kwargs['use_locking'], optimizer_kwargs['name'])
        self.optimizer_kwargs = optimizer_kwargs
        self.var_list = var_list

        self.lr = tf.Variable(self.optimizer_kwargs['learning_rate'], dtype=tf.float32, trainable=False, name='lr')
        with tf.name_scope('base_optimizer'):
            if self.optimizer_kwargs['base_optimizer'] == 'SGD':
                self.sgd_optim = tf.train.GradientDescentOptimizer(self.lr)
            elif self.optimizer_kwargs['base_optimizer'] == 'Adam':
                self.sgd_optim = tf.train.AdamOptimizer(self.lr, self.optimizer_kwargs['beta1'], self.optimizer_kwargs['beta2'])
            elif self.optimizer_kwargs['base_optimizer'] == 'Momentum':
                self.sgd_optim = tf.train.MomentumOptimizer(self.lr, self.optimizer_kwargs['momentum'], use_nesterov=False)
            elif self.optimizer_kwargs['base_optimizer'] == 'Nestrov':
                self.sgd_optim = tf.train.MomentumOptimizer(self.lr, self.optimizer_kwargs['momentum'], use_nesterov=True)
            elif self.optimizer_kwargs['base_optimizer'] == 'Adagrad':
                self.sgd_optim = tf.train.AdagradOptimizer(self.lr)
            elif self.optimizer_kwargs['base_optimizer'] == 'Adadelta':
                self.sgd_optim = tf.train.AdadeltaOptimizer(self.lr, self.optimizer_kwargs['rho'])
            else:
                raise Exception('Not implemented!')


    def dot(self, xs, ys):
        return tf.add_n([tf.reduce_sum(x * y) for x, y in zip(xs, ys)])

    def mult_list(self, xs, ys):
        return [x * y for x, y in zip(xs, ys)]

    def add_list(self, xs, ys):
        return [x + y for x, y in zip(xs, ys)]

    def assign_list(self, xs, ys):
        return [tf.assign(x, y) for x, y in zip(xs, ys)]

    def maximum_list(self, xs, y):
        return [tf.maximum(x, y) for x in xs]

    def minimum_list(self, xs, y):
        return [tf.minimum(x, y) for x in xs]

    def where_positive_list(self, masks, x, y):
        return [tf.where(m > 0, tf.ones_like(m) * x, tf.ones_like(m) * y) for m in masks]


    @lazy_property
    def v(self):
        if self.optimizer_kwargs['ignore_big_ones']:
            res = [tf.sign(var - self.snapshot_curr[var]) for var in self.var_list]
            res = self.maximum_list(res, -1.0)
            res = self.minimum_list(res, 1.0)
            return res

        return [var - self.snapshot_curr[var] for var in self.var_list]

    @lazy_property
    def u(self):
        if self.optimizer_kwargs['ignore_big_ones']:
            res = [tf.sign(self.snapshot_curr[var] - self.snapshot_prev[var]) for var in self.var_list]

            # res = self.maximum_list(res, -1.0)
            # res = self.minimum_list(res, 1.0)
            return res

        return [self.snapshot_curr[var] - self.snapshot_prev[var] for var in self.var_list]

    @lazy_property
    def v_norm(self):
        return tf.global_norm(self.v)

    @lazy_property
    def u_norm(self):
        return tf.global_norm(self.u)

    @lazy_property
    def lr_update_multiplier(self):
        with tf.name_scope('lr_update_multiplier'):
            if self.optimizer_kwargs['per_variable'] == False:
                return self.dot(self.v, self.u) / (self.u_norm*self.v_norm)
            else:
                return [tf.sign((var - self.snapshot_curr[var])*(self.snapshot_curr[var] - self.snapshot_prev[var])) for var in self.var_list]

    #1.
    @lazy_property
    def update_lr(self):
        def log2(x):
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
            return numerator / denominator



        with tf.name_scope('update_lr'):
            # 1. shift lr
            shift_lr = tf.assign(self.prev_lr, self.lr)

            with tf.control_dependencies([shift_lr]):
                #angle = self.lr_update_multiplier - self.angle_moving_avarage.average(self.lr_update_multiplier)
                #angle = self.lr_update_multiplier - self.angle_moving_avarage
                #angle = self.angle_moving_avarage
                angle = self.lr_update_multiplier
                # 2. update lr
                if self.optimizer_kwargs['update_rule'] == 'log':
                    factor = log2(1.0 + (1.0 + angle))
                elif self.optimizer_kwargs['update_rule'] == 'sqrt':
                    factor = (1.0 + angle)**0.5
                elif self.optimizer_kwargs['update_rule'] == 'slow_at_first':
                    i = tf.cast(self.sgd_counter/(2*self.optimizer_kwargs['iters_per_adjust']), tf.float32)
                    factor = (1.0 + angle)**(1.0/tf.maximum(float(self.optimizer_kwargs['n_epochs']) - i, 1.0))
                elif self.optimizer_kwargs['update_rule'] == 'divide_when_neg':
                    factor = tf.where(angle < 0, self.optimizer_kwargs['divide_factor'], 1.0)
                else:
                    factor = 1.0 + angle

                _update_lr = tf.assign(self.curr_lr, self.lr * factor)
            return tf.group(*[shift_lr, _update_lr, tf.assign(self.lr_slide_index, 0)])

    #2.
    @lazy_property
    def shift_snapshots(self):
        with tf.name_scope('shift_snapshots'):
            with tf.control_dependencies([]):
                return [tf.assign(self.snapshot_prev[var], self.snapshot_curr[var]) for var in self.var_list]

    #3.
    @lazy_property
    def take_snapshot_curr(self):
        with tf.name_scope('take_snapshot_curr'):
            with tf.control_dependencies(self.shift_snapshots):
                return [tf.assign(self.snapshot_curr[var], var) for var in self.var_list]



    def _create_slots(self, var_list):
        self.sgd_optim._create_slots(var_list)
        # Create slots for the first and second moments.
        self.snapshot_curr = {}
        self.snapshot_prev = {}

        for v in var_list:
            self._zeros_slot(v, "snapshot_prev", self._name)
            self._zeros_slot(v, "snapshot_curr", self._name)

        for v in var_list:
            self.snapshot_prev[v] = self.get_slot(v, "snapshot_prev")
            self.snapshot_curr[v] = self.get_slot(v, "snapshot_curr")




    def _prepare(self):
        self.sgd_optim._prepare()

        self.sgd_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name='sgd_counter')

        self.prev_lr = tf.Variable(self.optimizer_kwargs['learning_rate'], dtype=tf.float32, trainable=False, name='prev_lr')
        self.curr_lr = tf.Variable(self.optimizer_kwargs['learning_rate'], dtype=tf.float32, trainable=False, name='curr_lr')

        with tf.name_scope('v'):
            self.v
        with tf.name_scope('u'):
            self.u

        with tf.name_scope('angle'):
            self.angle_moving_avarage = tf.Variable(0.0, dtype=tf.float32, trainable=False)
            self.lr_update_multiplier

        ########## SLIDE LR ##########
        with tf.name_scope('slide_lr'):
            #self.lr_slide_index = self.sgd_counter % self.optimizer_kwargs['iters_per_adjust']
            self.lr_slide_index = tf.Variable(0, dtype=tf.int32, trainable=False)
            a = tf.cast(self.lr_slide_index, tf.float32) / self.optimizer_kwargs['iters_per_adjust']
            self.set_lr = tf.assign(self.lr, a * self.curr_lr + (1 - a) * self.prev_lr)

        ###### UPDATE LR #######
        with tf.name_scope('update_lr_cond'):
            should_update = tf.logical_and(tf.equal(self.sgd_counter % self.optimizer_kwargs['iters_per_adjust'], self.optimizer_kwargs['iters_per_adjust'] - 1),
                                           tf.less(2 * self.optimizer_kwargs['iters_per_adjust'] - 1, self.sgd_counter))

            #SV DEBUG
            self.update_lr_cond = tf.cond(pred=should_update, true_fn=lambda: self.update_lr, false_fn=lambda: tf.no_op())

        with tf.name_scope('maintain_averages_op_cond'):
            self.maintain_averages_op_cond = tf.cond(pred=(self.u_norm*self.v_norm > 0),
                                          true_fn=lambda: tf.group(*[self.angle_moving_avarage.assign(0.001*self.lr_update_multiplier + 0.999*self.angle_moving_avarage)]),
                                          false_fn=lambda: tf.no_op())


        with tf.name_scope('handle_snapshots'):
            self.update_snapshot_cond = tf.cond(pred=tf.equal(self.sgd_counter % self.optimizer_kwargs['iters_per_adjust'], 0),
                                                true_fn=lambda: tf.group(*(self.shift_snapshots + self.take_snapshot_curr)), false_fn=lambda: tf.no_op())




        ###### SUMMARY #######
        tf.summary.scalar('sgd_counter', self.sgd_counter, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        tf.summary.scalar('lr', self.lr, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        tf.summary.scalar('u_norm', self.u_norm, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        tf.summary.scalar('v_norm', self.v_norm, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        tf.summary.scalar('angle', tf.cond(self.u_norm*self.v_norm > 0, lambda: self.lr_update_multiplier, lambda: 0.0), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

        tf.summary.scalar('should_update', tf.cast(should_update, tf.int32), [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        tf.summary.scalar('lr_slide_index', self.lr_slide_index, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        tf.summary.scalar('target_lr', self.curr_lr, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

        tf.summary.scalar('angle_moving_avarage', self.angle_moving_avarage, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])

        tf.summary.scalar('bias_corrected_angle', self.lr_update_multiplier - self.angle_moving_avarage, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])
        #### DEBUG ######
        #tf.summary.scalar('a', a, [SgdAdjustOptimizer.SUMMARY_SGD_KEY])


        self.sgd_summaries = tf.summary.merge_all(SgdAdjustOptimizer.SUMMARY_SGD_KEY)


    def _apply_dense(self, grad, var):
        return self.sgd_optim._apply_dense(grad, var)

    def _finish(self, update_ops, name_scope):
        res = [self.sgd_optim._finish(update_ops, name_scope)]

        with tf.control_dependencies(res):
            #SV DEBUG
            res.append(self.update_lr_cond)
            res.append(self.maintain_averages_op_cond)
            with tf.control_dependencies(res):
                res.append(self.set_lr)
                res.append(self.update_snapshot_cond)
                with tf.control_dependencies(res):
                    res.append(self.sgd_counter.assign_add(1))
                    res.append(self.lr_slide_index.assign_add(1))


        return tf.group(*res)






