
import tensorflow as tf
import numpy as np

from sesop_optimizer import SubspaceOptimizer

from tf_utils import avarge_on_feed_dicts, avarge_n_calls

# Usage:
# ----------
# run_epoch
# run_sesop

# run_epoch
# run_sesop
class SeboostOptimizer:
    SUMMARY_SGD_KEY = 'sgd_debug_summaries'

    def set_summary_writer(self, writer):
        if self.history_size > 0:
            self.subspace_optim.set_summary_writer(writer)
        self.writer = writer

    def __init__(self, loss, batch_provider, var_list, history_size, **optimizer_kwargs):
        initial_lr = optimizer_kwargs['lr']
        self.lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
        self.curr_lr = initial_lr
        self.lr_placeholder = tf.placeholder(dtype=tf.float32)
        self.update_lr = tf.assign(self.lr, self.lr_placeholder)
        self.input = batch_provider.batch()
        self.bp = batch_provider
        self.loss = loss
        self.history_size = history_size
        self.best_loss = 10000.0


        self.batch_size = optimizer_kwargs['batch_size']
        self.train_dataset_size = optimizer_kwargs['train_dataset_size']
        self.adaptable_learning_rate = optimizer_kwargs['adaptable_learning_rate']
        self.num_of_batches_per_sesop = optimizer_kwargs['num_of_batches_per_sesop']



        with tf.name_scope('seboost_optimizer'):
            if optimizer_kwargs['seboost_base_method'] == 'SGD':
                self.sgd_optim = tf.train.GradientDescentOptimizer(self.lr)
            else:
                #implement others here
                assert (False)

            self.sgd_op = self.sgd_optim.minimize(loss, var_list=var_list)
            #self.sgd_grads = self.sgd_optim.compute_gradients(loss)
            self.sgd_grads = tf.gradients(loss, var_list)

            self.full_loss_summary = tf.summary.scalar('full_loss', loss, ['full_loss'])
            tf.summary.scalar('loss_during_sgd', loss, [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('weights_norm_during_sgd', tf.global_norm(var_list), [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('grad_norm_norm_during_sgd', tf.global_norm(self.sgd_grads), [SeboostOptimizer.SUMMARY_SGD_KEY])
            tf.summary.scalar('learning_rate', self.lr, [SeboostOptimizer.SUMMARY_SGD_KEY])

            if history_size > 0:
                with tf.name_scope('sesop_optimizer'):
                    self.subspace_optim = SubspaceOptimizer(loss, var_list, history_size, **optimizer_kwargs)

        self.sgd_summaries = tf.summary.merge_all(SeboostOptimizer.SUMMARY_SGD_KEY)

        self.explotion_counter = 0
        self.iter = 0
        self.steps_since_last_sesop_update = 0

    def run_sesop(self, sess):
        if self.history_size == 0:
            self.iter += 1
            self.steps_since_last_sesop_update += 1
            return

        self.bp.set_deque_batch_size(sess, self.train_dataset_size/2)
        self.writer.add_summary(sess.run(self.full_loss_summary, {self.loss : avarge_n_calls(sess, self.loss, 2)}), 2 * self.iter)
        self.bp.set_deque_batch_size(sess, self.batch_size)
        #tf.tensorboard.tensorboard
        feed_dicts = []
        for z in range(self.num_of_batches_per_sesop):
            _x, _labels = sess.run(self.input)
            feed_dicts.append({self.input[0]: _x, self.input[1]: _labels})

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
            #if SESOP corrected us, we update the lr
            if _distance_sesop_moved > 1e-5:
                # model: iters*lr*redundancy = sgd_distance
                # where redundancy is between 0 and 1, and it means, what is the precentage of distance we "lose" in an epoch
                # Lets plot redundancy = sgd_distance/(iters*lr)
                # We will then set lr = desigred_distance/(iters*redundancy)

                #self.curr_lr = _distance_seboost_moved/(float(self.train_dataset_size/self.batch_size)*redundancy)

                #self.curr_lr = _distance_seboost_moved/(float(self.train_dataset_size/self.batch_size)*redundancy)

                next_lr = self.curr_lr * (1.0 + _sgd_sesop_dot_product/(_sgd_distance_moved*_sgd_distance_moved))
                if next_lr > 0:
                    self.curr_lr = next_lr
                    print 'setting lr to: ' + str(self.curr_lr)
                    sess.run(self.update_lr, {self.lr_placeholder: self.curr_lr})
                else:
                    print 'not setting lr: ' + str(next_lr)

        self.bp.set_deque_batch_size(sess, self.train_dataset_size/2)
        self.writer.add_summary(sess.run(self.full_loss_summary, {self.loss: avarge_n_calls(sess, self.loss, 2)}), 2 * self.iter + 1)
        self.bp.set_deque_batch_size(sess, self.batch_size)

        print 'loss after = ' + str(avarge_on_feed_dicts(sess, [self.loss], feed_dicts)[0])
        print '------------------'

        self.iter += 1

    def run_epoch(self, sess):
        for i in range(self.train_dataset_size/self.batch_size):
            s = sess.run(self.sgd_summaries)
            self.writer.add_summary(s, self.iter * (self.train_dataset_size/self.batch_size) + i)
            sess.run(self.sgd_op)
            self.steps_since_last_sesop_update += 1


