

import tensorflow as tf
import numpy as np
#import argparse
import sys
import threading
from threading import current_thread

import experiments_manager
from summary_manager import SummaryManager
from tf_utils import *



class HVar:
    #this contains all alphas in the graph
    all_hvars = {}

    @classmethod
    def reset(cls):
        HVar.all_hvars[current_thread()] = []

    #problem is when someone is doing get_variable, he expect to reuse the same variable
    #So we should pass this class a ready variable
    #def __init__(self, initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None):
    #    var = tf.Variable(initial_value, trainable, collections, validate_shape, caching_device, name, variable_def, dtype, expected_shape, import_scope)

    def __init__(self, var):
        #self.name = var.name.split(":")[0].split("/")[-1]
        self.name = var.name.split(":")[0]
        self.var = var
        with tf.variable_scope(self.name):
            self.sub_init(var)

    def sub_init(self, var):
        experiment = experiments_manager.ExperimentsManager.get().get_current_experiment()
        self.hSize = experiment.getFlagValue('hSize')
        self.nodes = 1 #experiment.getFlagValue('nodes')
        self.node_id = 0 #experiment.getFlagValue('node_id')

        self.var = var

        self.history = []
        self.history_aplha = []

        self.replicas = []
        self.replicas_aplha = []

        self.next_idx = 0
        self.op_cache = {}
        self.o = None



        with tf.name_scope(self.name + '_subspace'):
            # snapshot is taken after each sesop. after a sesop, the snapshot will contain the value after sesop ran.
            # we need this variable to be shared so we will be able to push the "after sesop value" back into the workers.
            self.last_snapshot = tf.get_variable(initializer=var.initialized_value(), name='snapshot')

            #node 0 also holds 'nodes - 1' variables. 1 for each other node:
            #the last n-1 replicas are "nodes" replicas.
            with tf.name_scope('replicas'):
                for i in range(self.nodes - 1):
                    #Note we use tf.getVariable, so thes variables are shared between all nodes.
                    self.replicas.append(tf.get_variable(initializer=np.zeros(var.get_shape()), \
                                                        dtype=var.dtype.base_dtype, name='replica_' + str(i)))

            if self.node_id == 0:
                with tf.name_scope('history'):
                    for i in range(self.hSize):
                        self.history.append(tf.Variable(np.zeros(var.get_shape()),\
                            dtype=var.dtype.base_dtype, name='h_' + str(i)))

                with tf.name_scope('history_alpha'):
                    for i in range(self.hSize):
                        self.history_aplha.append(tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_h_' + str(i)))
                        SummaryManager.get().add_iter_summary(tf.summary.histogram('alphas_h', self.history_aplha[-1]))

                with tf.name_scope('replicas_aplha'):
                    for i in range(self.nodes - 1):
                        self.replicas_aplha.append(tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_n_' + str(i)))
                        SummaryManager.get().add_iter_summary(tf.summary.histogram('alphas_n', self.replicas_aplha[-1]))


        for i in range(self.hSize):
            self.push_history_op() #make sure all ops are created

        if current_thread() not in HVar.all_hvars:
            HVar.all_hvars[current_thread()] = []

        HVar.all_hvars[current_thread()].append(self)
        assert(self.next_idx == 0)

    def out(self):
        if self.o is not None:
            return self.o

        with tf.name_scope(self.name + '_out'):
            #return an affine combination of the history vectors
            #and a dictonary to add to feed_dict.
            self.o = self.var

            if self.node_id == 0:
                for r, a in zip(self.history, self.history_aplha):
                    self.o += r * a

                for r, a in zip(self.replicas, self.replicas_aplha):
                    self.o += r*a

            return self.o

    # create an op that puts var of this node into its replica
    #Called before sesop!
    def push_to_master_op(self):
        assert (self.node_id != 0)
        return tf.assign(self.replicas[self.node_id - 1], self.out())

    # this must be called after sesop was executed!
    #Called after sesop!
    def pull_from_master_op(self):
        assert (self.node_id != 0)
        return tf.assign(self.var, self.last_snapshot)

    # returns an op that updates history and snapshot (executed after optimization on alpha)
    #This must be called when alpahs are non zeros!!!
    def push_history_op(self):
        if self.next_idx not in self.op_cache:
            #print 'HVar Cache Miss, creating the op for var ' + str(self.var.name) + ', idx = ' + str(self.next_idx)
            sys.stdout.flush()
            with tf.name_scope(self.name + '_update'):

                #first we update the original variable to the sesop result
                update_var_op = tf.assign(self.var, self.out())
                with tf.control_dependencies([update_var_op]):
                    #now we update the history (self.var contain the sesop result):
                    update_history_op = tf.assign(self.history[self.next_idx], self.var - self.last_snapshot)
                    with tf.control_dependencies([update_history_op]):
                        #now we update the last_snapshot to be the sesop result
                        update_snapshot_op = tf.assign(self.last_snapshot, self.var)
                        with tf.control_dependencies([update_snapshot_op]):
                            #finally we reset all the alphas (infact we can take this out of the dependecy)
                            #as it only affect self.out()
                            reset_alpha_op = self.zero_alpha_op()
                            self.op_cache[self.next_idx] =\
                                tf.group(update_history_op, update_var_op, update_snapshot_op, reset_alpha_op)

        old_idx = self.next_idx
        self.next_idx = (self.next_idx + 1)%self.hSize

        return self.op_cache[old_idx]

    def zero_alpha_op(self):
        group_op = tf.no_op()
        for a in self.replicas_aplha + self.history_aplha:
            group_op = tf.group(group_op, tf.assign(a, np.zeros(1)))
        return group_op




    #the alphas from sesop (the coefitients that choose the history vector)
    @classmethod
    def all_trainable_alphas(self):
        alphas = []
        for hvar in HVar.all_hvars[current_thread()]:
            alphas.extend(hvar.replicas_aplha + hvar.history_aplha)
        return alphas

    #all the regular weights to be trained
    @classmethod
    def all_trainable_weights(self):
        weights = []
        for hvar in HVar.all_hvars[current_thread()]:
            weights.append(hvar.var)
        return weights

    @classmethod
    def all_history_update_ops(self):
        group_op = tf.no_op()
        for hvar in HVar.all_hvars[current_thread()]:
            group_op = tf.group(group_op, hvar.push_history_op())

        return group_op


class SeboostOptimizer:


    #batched_input, batched_labels are tensors that prodece batches
    #is_training is a tensor that will be true while training and false while testing
    #we run CG once in sesop_freq iterations
    def __init__(self, loss, batched_input, batched_labels):

        self.break_chains = True
        self.loss = loss
        self.iteration_ran = 0
        self.sesop_iteration_ran = 0
        self.avg_gain_from_cg = 0.0
        self.iter_summaries = SummaryManager.get().merge_iters()

        self.experiment = experiments_manager.ExperimentsManager.get().get_current_experiment()
        lr = self.experiment.getFlagValue('lr')
        self.sesop_freq = self.experiment.getFlagValue('sesop_freq')

        self.sgd_steps = int(1 / self.sesop_freq)
        self.dataset_size = self.experiment.getFlagValue('dataset_size')
        self.iters_per_epoch = self.dataset_size / self.experiment.getFlagValue('b')


        # sesop_offset_in_epoch is the number of sgd iterations we need to do before we run sesop
        # (counting from the begining of an epoch)
        self.sesop_offset_in_epoch = self.sgd_steps % self.iters_per_epoch
        self.num_of_epoch_without_sesop = self.sgd_steps / self.iters_per_epoch
        self.num_of_sesops_in_epoch = self.iters_per_epoch/self.sgd_steps
        self.epochs_ran_so_far = 0
        self.sesop_runs = 0



        self.sgd_ran_times_tf = tf.Variable(0, dtype=tf.int32)
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, name='minimizer', \
                var_list=HVar.all_trainable_weights(), global_step=self.sgd_ran_times_tf)

        if self.break_chains == False:
            with tf.name_scope('sesop_offset_train_steps'):
                self.sesop_offset_train_steps = chain_training_step(lambda: tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,\
                name='minimizer', var_list=HVar.all_trainable_weights(), global_step=self.sgd_ran_times_tf), self.sesop_offset_in_epoch)

            with tf.name_scope('epoch_train_steps'):
                self.epoch_train_steps = chain_training_step(lambda: tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,\
                name='minimizer', var_list=HVar.all_trainable_weights(), global_step=self.sgd_ran_times_tf), self.iters_per_epoch)

        with tf.name_scope('10_train_steps'):
            self.ten_train_steps = chain_training_step(lambda: tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,\
            name='minimizer', var_list=HVar.all_trainable_weights(), global_step=self.sgd_ran_times_tf), 10)

        with tf.name_scope('epoch_mod_10_train_steps'):
            self.epoch_mod_10_train_steps = chain_training_step(lambda: tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,\
            name='minimizer', var_list=HVar.all_trainable_weights(), global_step=self.sgd_ran_times_tf), self.iters_per_epoch%10)

        with tf.name_scope('sesop_offset_mod_10_train_steps'):
            self.sesop_offset_mod_10_train_steps = chain_training_step(lambda: tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,\
            name='minimizer', var_list=HVar.all_trainable_weights(), global_step=self.sgd_ran_times_tf), self.sesop_offset_in_epoch%10)


        self.cg_var_list = HVar.all_trainable_alphas()
        self.cg = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=self.cg_var_list,\
            method='CG', options={'maxiter':5})

        #all_trainable_weights
        self.batched_input, self.batched_labels = batched_input, batched_labels



        print 'self.num_of_epoch_without_sesop = ' + str(self.num_of_epoch_without_sesop)
        print 'self.num_of_sesops_in_epoch = ' + str(self.num_of_sesops_in_epoch)
        print 'self.sesop_offset_in_epoch = ' + str(self.sesop_offset_in_epoch)

    def run_sesop(self, sess, _feed_dict):
        batched_input_actual, batched_labels_actual = \
            sess.run([self.batched_input, self.batched_labels], feed_dict=_feed_dict)

        sesop_feed_dict = {self.batched_input: batched_input_actual,
                           self.batched_labels: batched_labels_actual}
        sesop_feed_dict.update(_feed_dict)

        total_count_sgd_steps = sess.run(self.sgd_ran_times_tf)
        print 'Running sesop when self.sgd_ran_times_tf = ' + str(total_count_sgd_steps)
        print 'So actual sesop freq is: ' + str(float(self.sesop_runs + 1) / total_count_sgd_steps)

        # now run sesop:
        self.cg.minimize(sess, feed_dict=sesop_feed_dict)
        self.sesop_runs += 1
        return sess.run(self.loss, feed_dict=sesop_feed_dict)

    def run_epoch_sgd_steps(self, sess, _feed_dict):
        if self.break_chains:
            for i in range(self.iters_per_epoch/10):
                _, loss = sess.run([self.ten_train_steps, self.loss], feed_dict=_feed_dict)
            _, loss = sess.run([self.epoch_mod_10_train_steps, self.loss], feed_dict=_feed_dict)
            return loss

        _, loss = sess.run([self.epoch_train_steps, self.loss], feed_dict=_feed_dict)
        return loss


    def run_sesop_offset_sgd_steps(self, sess, _feed_dict):
        if self.break_chains:
            for i in range(self.sesop_offset_in_epoch/10):
                _, loss = sess.run([self.ten_train_steps, self.loss], feed_dict=_feed_dict)
            _, loss = sess.run([self.sesop_offset_mod_10_train_steps, self.loss], feed_dict=_feed_dict)
            return loss


        _, loss = sess.run([self.sesop_offset_train_steps, self.loss], feed_dict=_feed_dict)
        return loss

    def run_epoch(self, sess, _feed_dict):
        #we have 2 full epochs without sesop
        #Then in the begining of the first
        #should we run a full epoch without sesop?
        if self.experiment.getFlagValue('hSize') == 0 or self.epochs_ran_so_far < self.num_of_epoch_without_sesop:
            loss = self.run_epoch_sgd_steps(sess, _feed_dict)
            total_count_sgd_steps = sess.run(self.sgd_ran_times_tf)
            print 'Ran = ' + str(total_count_sgd_steps) + ' sgd steps'
            #print 'loss = ' + str(loss)
            self.epochs_ran_so_far += 1
            return loss

        print 'Running SESOP!'
        self.epochs_ran_so_far = 0
        batched_input_actual, batched_labels_actual = \
            sess.run([self.batched_input, self.batched_labels], feed_dict=_feed_dict)

        sesop_feed_dict = {self.batched_input: batched_input_actual,
                           self.batched_labels: batched_labels_actual}
        sesop_feed_dict.update(_feed_dict)

        #sesop is done once every few epochs, and now is the time.
        if self.num_of_sesops_in_epoch == 0:
            return self.run_sesop(sess, _feed_dict)

        for i in range(self.num_of_sesops_in_epoch):
            #now run the rest of the steps
            loss = self.run_sesop_offset_sgd_steps(sess, _feed_dict)
            #now run sesop
            loss = self.run_sesop(sess, _feed_dict)

        return loss

"""
    def is_sesop_iteration(self):
        return self.iteration_ran%self.sgd_steps == 0 and len(self.cg_var_list) != 0


    #_feed_dict is the feed_dict needed to run regular sgd iteration
    #sesop_feed_dict should contain feeds for the batch sesop will use!
    #return a list of train_loss. The last elment in the list contain the loss after sesop.
    def run_sesop_iteration(self, sess, _feed_dict, sesop_feed_dict):
        #run sesop_freq SGD iterations:
        if True or not self.is_sesop_iteration():
            _, loss = sess.run([self.train_step, self.loss], feed_dict=_feed_dict)
            self.iteration_ran += 1

            self.train_loss.append(loss)
            self.writer.add_summary(sess.run(self.iter_summaries, feed_dict=_feed_dict), self.iteration_ran)
            #return loss


        self.loss_before_sesop.append(sess.run(self.loss, feed_dict=sesop_feed_dict))
        #run 1 CG iteration
        #print 'sess = ' + str(sess)
        #print 'sesop_feed_dict = ' + str(sesop_feed_dict)
        self.cg.minimize(sess, feed_dict=sesop_feed_dict)
        self.iteration_ran += 1
        self.sesop_iteration_ran += 1

        self.loss_after_sesop.append(sess.run(self.loss, feed_dict=sesop_feed_dict))

        self.avg_gain_from_cg += self.loss_before_sesop[-1] - self.loss_after_sesop[-1]
        #print 'Gain from CG: ' + str(self.avg_gain_from_cg/(self.sesop_iteration_ran))
        sys.stdout.flush()
        self.train_loss.append(self.loss_after_sesop[-1])

        #We want to capture the values of alpha before we zero them, so we need to call
        #the summary before we zero them in self.history_update_ops
        self.writer.add_summary(sess.run(self.iter_summaries, feed_dict=_feed_dict), self.iteration_ran)

        #Now when alphas are optimized, run the update history ops:
        sess.run(HVar.all_history_update_ops())




        return self.loss_after_sesop[-1]
"""



