
import tensorflow as tf
import numpy as np

from sesop_optimizer import SubspaceOptimizer

from tf_utils import avarge_on_feed_dicts, avarge_n_calls
from lr_auto_adjust_sgd_optimizer import SgdAdjustOptimizer

# Usage:
# ----------
# run_epoch
# run_sesop

# run_epoch
# run_sesop
class SeboostOptimizer:
    SUMMARY_SGD_KEY = 'sgd_debug_summaries'

    def is_sesop_on(self):
        if self.history_size <= 0 and not self.optimizer_kwargs['use_grad_dir']:
            return False

        if 'disable_sesop_at_all' in self.optimizer_kwargs and self.optimizer_kwargs['disable_sesop_at_all'] == True:
            return False

        return True


    def set_summary_writer(self, writer):
        if self.is_sesop_on():
            self.subspace_optim.set_summary_writer(writer)
        self.writer = writer

    def update_lr_op_per_var(self, sess, _sgd_distance_moved, _distance_sesop_moved, _distance_seboost_moved, _sgd_sesop_dot_product):
        for i in range(len(self.orig_var_list)):
            var = self.orig_var_list[i]
            _sgd = _sgd_distance_moved[i]
            _sesop = _distance_sesop_moved[i]
            _seboost = _distance_seboost_moved[i]
            _sgd_sesop_dot = _sgd_sesop_dot_product[i]

            curr_lr = sess.run(self.lr_mult_factor[var])

            #1e-30 is here just to make sure 0/0 is 0.
            next_lr = curr_lr * abs(1.0 + _sgd_sesop_dot / (_sgd * _sgd + 1e-30))
            next_lr = np.where(_sesop > 1e-6, next_lr, curr_lr)

            #untrusted_idx = (_sesop < 1e-6).astype(int)

            print 'setting lr of var ' + str(var) + ' to (max): ' + str(np.amax(next_lr))
            sess.run(self.update_lr_mult_factor_op[var], { self.lr_mult_factor_placeholder[var]: next_lr })

    def __init__(self, loss, batch_provider, var_list, history_size, **optimizer_kwargs):
        initial_lr = optimizer_kwargs['lr']
        self.lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)

        #
        if optimizer_kwargs['per_variable']:
            self.lr_mult_factor = { var :tf.Variable(np.ones(var.get_shape()), dtype=var.dtype.base_dtype, trainable=False) for var in var_list }
            self.lr_mult_factor_placeholder = { var : tf.placeholder(dtype=tf.float32) for var in var_list }
            self.update_lr_mult_factor_op = { var : tf.assign(self.lr_mult_factor[var], self.lr_mult_factor_placeholder[var]) for var in var_list }

        self.curr_lr = initial_lr
        self.avg_lr = 0.0

        self.lr_placeholder = tf.placeholder(dtype=tf.float32)
        self.update_lr_op = tf.assign(self.lr, self.lr_placeholder)
        self.input = batch_provider.batch()
        self.bp = batch_provider
        self.loss = loss
        self.history_size = history_size
        self.best_loss = 10000.0
        self.orig_var_list = var_list

        self.optimizer_kwargs = optimizer_kwargs

        self.batch_size = optimizer_kwargs['batch_size']
        self.train_dataset_size = optimizer_kwargs['train_dataset_size']
        self.adaptable_learning_rate = optimizer_kwargs['adaptable_learning_rate']
        self.num_of_batches_per_sesop = optimizer_kwargs['num_of_batches_per_sesop']
        self.sesop_private_dataset = optimizer_kwargs['sesop_private_dataset']
        self.weight_decay = optimizer_kwargs['weight_decay']
        self.global_step = tf.Variable(0, dtype=tf.int64)

        with tf.name_scope('seboost_optimizer'):
            if optimizer_kwargs['seboost_base_method'] == 'SGD':
                self.sgd_optim = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer_kwargs['seboost_base_method'] == 'SGD_adjust':
                optimizer_kwargs['learning_rate'] = initial_lr
                optimizer_kwargs['use_locking'], optimizer_kwargs['name'] = False, 'SgdAdjustOptimizer'

                self.sgd_optim = SgdAdjustOptimizer(var_list, **optimizer_kwargs)
            elif optimizer_kwargs['seboost_base_method'] == 'Adam':
                self.sgd_optim = tf.train.AdamOptimizer(self.lr, optimizer_kwargs['beta1'], optimizer_kwargs['beta2'])
            elif optimizer_kwargs['seboost_base_method'] == 'Adagrad':
                self.sgd_optim = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=optimizer_kwargs['initial_accumulator_value'])
            elif optimizer_kwargs['seboost_base_method'] == 'AdagradDA':
                self.sgd_optim = tf.train.AdagradDAOptimizer(self.lr, self.global_step,
                                                             initial_gradient_squared_accumulator_value=optimizer_kwargs['initial_accumulator_value'],
                                                             l2_regularization_strength=self.weight_decay)
            elif optimizer_kwargs['seboost_base_method'] == 'Adadelta':
                self.sgd_optim = tf.train.AdadeltaOptimizer(self.lr, optimizer_kwargs['rho'])
            elif optimizer_kwargs['seboost_base_method'] == 'Momentum':
                self.sgd_optim = tf.train.MomentumOptimizer(self.lr, optimizer_kwargs['momentum'])
            else:
                raise Exception('Not implemented!')


            costs = []
            new_loss = loss
            # SV ARTICLE
            if optimizer_kwargs['seboost_base_method'] != 'AdagradDA':
                for var in var_list:
                    if not (var.op.name.find(r'bias') > 0):
                        costs.append(tf.nn.l2_loss(var))
                new_loss += (tf.add_n(costs) * self.weight_decay)

            grads_and_vars = self.sgd_optim.compute_gradients(new_loss, var_list=var_list)
            self.sgd_grads = [g for g, v in grads_and_vars]

            if optimizer_kwargs['per_variable']:
                grads_and_vars = [(g*self.lr_mult_factor[v], v) for g,v in grads_and_vars]
            self.sgd_op = self.sgd_optim.apply_gradients(grads_and_vars)



            self.test_full_loss_summary = tf.summary.scalar('test_full_loss', loss, ['test_full_loss'])
            self.full_loss_summary = tf.summary.scalar('full_loss', loss, ['full_loss'])
            tf.summary.scalar('regularized_loss_during_sgd', new_loss, [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('loss_during_sgd', loss, [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('weights_norm_during_sgd', tf.global_norm(var_list), [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('grad_norm_norm_during_sgd', tf.global_norm(self.sgd_grads), [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('learning_rate', self.lr, [SeboostOptimizer.SUMMARY_SGD_KEY])

            if self.is_sesop_on():
                with tf.name_scope('sesop_optimizer'):
                    self.subspace_optim = SubspaceOptimizer(loss, var_list, history_size, **optimizer_kwargs)

        self.sgd_summaries = tf.summary.merge_all(SeboostOptimizer.SUMMARY_SGD_KEY)
        self.sgd_summary_idx = 0
        self.sesop_summary_idx = 0
        self.lr_change_idx = 0
        self.explotion_counter = 0
        self.iter = 0

    def run_sesop(self, sess):
        if not self.is_sesop_on():
            return

        self.write_loss_messures(sess, self.sesop_summary_idx)

        if self.sesop_private_dataset:
            self.bp.set_data_source(sess, 'sesop')

        #tf.tensorboard.tensorboard
        if self.optimizer_kwargs['break_sesop_batch']:
            feed_dicts = []
            for z in range(self.num_of_batches_per_sesop):
                _x, _labels = sess.run(self.input)
                feed_dicts.append({self.input[0]: _x, self.input[1]: _labels})
        else:
            self.bp.set_deque_batch_size(sess, self.num_of_batches_per_sesop*self.batch_size)
            _x, _labels = sess.run(self.input)
            print 'shape of sesop batch = ' + str(_x.shape)
            feed_dicts = [{self.input[0]: _x, self.input[1]: _labels}]
            self.bp.set_deque_batch_size(sess, self.batch_size)

        if self.sesop_private_dataset:
            self.bp.set_data_source(sess, 'train')

        curr_loss = avarge_on_feed_dicts(sess, [self.loss], feed_dicts)[0]
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss

        print 'self.iter = ' + str(self.iter)
        print 'loss before = ' + str(curr_loss)

        #Early stopping in case of divegence
        if curr_loss > 100*self.best_loss or np.isnan(curr_loss):
            self.explotion_counter += 1
            if self.explotion_counter > 5:
                raise Exception('Optimization has exploded, stoping early!')
        else:
            self.explotion_counter = 0

        _sgd_distance_moved, _distance_sesop_moved, _distance_seboost_moved, _sgd_sesop_dot_product = self.subspace_optim.minimize(session=sess, feed_dicts=feed_dicts)

        #redundancy = _sgd_distance_moved/(float(self.train_dataset_size/self.batch_size)*self.curr_lr)
        if self.adaptable_learning_rate == True:
            if self.optimizer_kwargs['per_variable']:
                self.update_lr_op_per_var(sess, _sgd_distance_moved, _distance_sesop_moved, _distance_seboost_moved, _sgd_sesop_dot_product)
            else:
                print '_sgd_sesop angle = ' + str(_sgd_sesop_dot_product/(_distance_sesop_moved*_sgd_distance_moved))
                factor = (1.0 + _sgd_sesop_dot_product / (_sgd_distance_moved*_sgd_distance_moved))
                # factor = factor**(1.0/(self.lr_change_idx + 1))
                # factor = factor*(self.lr_change_idx + 2)/(self.lr_change_idx + 1)
                # factor = factor ** (1.0 / (1 + (self.lr_change_idx) % 10) )

                #factor in [0, 2]

                #if SESOP corrected us, we update the lr
                # if _sgd_sesop_dot_product > 0:
                #     factor = 1.1
                # elif _sgd_sesop_dot_product < 0:
                #     factor = 0.9

                #factor = abs(1.0 + _sgd_sesop_dot_product/(_sgd_distance_moved*_sgd_distance_moved))

                print 'factor: ' + str(factor)
                #print 'np.log(self.lr_change_idx + 2.719): ' + str(np.log(self.lr_change_idx + 2.719))

                #the factor can not be bigger than the proportion of _distance_sesop_moved to _sgd_distance_moved
                #factor = min(factor, 1 + _sgd_distance_moved/_distance_sesop_moved)
                #factor = min(factor, 1.0 + 1.0 / (self.sesop_summary_idx + 1.0)**0.5)


                #factor = min(np.log(self.lr_change_idx + 2.719), factor)
                factor = max([factor, 0.5])
                factor = min([factor, 2.0])

                print 'factor after clamping ' + str(factor)
                # factor = min([factor, 1.1])
                # factor = max([factor, 0.9])

                if factor <= 2:
                    # model: iters*lr*redundancy = sgd_distance
                    # where redundancy is between 0 and 1, and it means, what is the precentage of distance we "lose" in an epoch
                    # Lets plot redundancy = sgd_distance/(iters*lr)
                    # We will then set lr = desigred_distance/(iters*redundancy)

                    #self.curr_lr = _distance_seboost_moved/(float(self.train_dataset_size/self.batch_size)*redundancy)

                    #self.curr_lr = _distance_seboost_moved/(float(self.train_dataset_size/self.batch_size)*redundancy)

                    #self.avg_lr = (self.avg_lr * self.lr_change_idx + self.curr_lr) / (self.lr_change_idx + 1)
                    #next_lr = self.avg_lr * factor

                    next_lr = self.curr_lr * factor

                    # self.avg_lr = (lr1 + ... + lrk)/k
                    # we want to change it to:
                    #(lr1 + ... + lrk + next_lr) / (k + 1)



                    self.curr_lr = next_lr
                    print 'setting lr to: ' + str(self.curr_lr)
                    sess.run(self.update_lr_op, {self.lr_placeholder: self.curr_lr})
                    self.lr_change_idx += 1


        self.write_loss_messures(sess, self.sesop_summary_idx + 1)
        self.sesop_summary_idx += 1
        print 'loss after = ' + str(avarge_on_feed_dicts(sess, [self.loss], feed_dicts)[0])
        print '------------------'


    def write_loss_messures(self, sess, index):
        self.bp.set_deque_batch_size(sess, self.train_dataset_size/2)
        self.writer.add_summary(sess.run(self.full_loss_summary, {self.loss: avarge_n_calls(sess, self.loss, 2)}), index)

        self.bp.set_data_source(sess, 'test')
        self.bp.set_deque_batch_size(sess, self.bp.test_size())
        self.writer.add_summary(sess.run(self.test_full_loss_summary, {self.loss : avarge_n_calls(sess, self.loss, 1)}), index)

        self.bp.set_data_source(sess, 'train')
        self.bp.set_deque_batch_size(sess, self.batch_size)

    def run_iter_without_sesop(self, sess):
        s = sess.run(self.sgd_summaries)
        self.writer.add_summary(s, self.sgd_summary_idx)
        self.sgd_summary_idx += 1
        sess.run(self.sgd_op)
        self.iter += 1

    def run_iter(self, sess):

        if (self.iter + 1) % (self.optimizer_kwargs['iters_per_sesop'] + 1) == 0 and self.is_sesop_on():
            self.run_sesop(sess)
        else:
            s = sess.run(self.sgd_summaries)
            self.writer.add_summary(s, self.sgd_summary_idx)
            self.sgd_summary_idx += 1
            sess.run(self.sgd_op)

        self.iter += 1


    #
    #
    #
    # def run_sgd_iters(self, sess, n):
    #     for i in range(n):
    #         s = sess.run(self.sgd_summaries)
    #         self.writer.add_summary(s, self.sgd_summary_idx)
    #         self.sgd_summary_idx += 1
    #         sess.run(self.sgd_op)
    #
    #
    # def run_epoch(self, sess):
    #     self.run_sgd_iters(sess, self.train_dataset_size/self.batch_size)
    #
    #

