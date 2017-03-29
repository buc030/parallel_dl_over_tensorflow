

import tensorflow as tf
import numpy as np
#import argparse
import sys
import threading
from threading import current_thread

import experiments_manager
from summary_manager import SummaryManager
from tf_utils import *



class SeboostOptimizerParams:
    def __init__(self, model):

        self.lr = model.experiment.getFlagValue('lr')
        self.sgd_steps = int(1.0/model.experiment.getFlagValue('sesop_freq'))
        self.sesop_batch_size = model.experiment.getFlagValue('sesop_batch_size')
        self.batch_size = model.experiment.getFlagValue('b')
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(model.loss(), name='minimizer', \
            var_list=model.hvar_mgr.all_trainable_weights())

        self.cg_var_list = model.hvar_mgr.all_trainable_alphas()
        self.cg = tf.contrib.opt.ScipyOptimizerInterface(model.loss(), var_list=self.cg_var_list, \
                                                         method='CG', options={'maxiter': 5})



#This runs the same optimization process for 'models' with corosponding metaparameters defined in 'experiments'
#All the models see the exact same data!
class SeboostOptimizer:

    #batched_input, batched_labels are tensors that prodece batches
    #is_training is a tensor that will be true while training and false while testing
    #we run CG once in sesop_freq iterations
    def __init__(self, experiments):
        self.sesop_runs = 0
        self.dataset_size = experiments[0].getFlagValue('dataset_size')
        self.batch_size = experiments[0].getFlagValue('b')
        self.models = []
        self.experiments = experiments

        for e in experiments:
            for m in e.models:
                self.models.append(m)


        bar = progressbar.ProgressBar(maxval=len(self.models), \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

        bar.start()
        self.params = {}
        for m in self.models:
            self.params[m] = SeboostOptimizerParams(m)
            bar.update(len(self.params))
        bar.finish()

        self.losses = [m.loss() for m in self.models]
        self.train_steps = [p.train_step for p in self.params.values()]

        self.curr_iter = 1

    def run_epoch(self, sess):

        bar = progressbar.ProgressBar(maxval=self.dataset_size/self.batch_size, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), progressbar.ETA()])
        bar.start()

        for i in range(self.dataset_size/self.batch_size):

            self.run_iter(sess)
            bar.update(i + 1)

        bar.finish()

        print 'Actual sesop freq is: ' + str(float(self.sesop_runs + 1) / (self.curr_iter + 1))


    def run_iter(self, sess):
        if self.curr_iter % self.params.values()[0].sgd_steps != 0 or \
                (self.models[0].experiment.getFlagValue('hSize') == 0 and self.models[0].experiment.getFlagValue('nodes') == 1):
            _, losses = sess.run([self.train_steps, self.losses])
            self.curr_iter += 1
            return None  # TODO: need to return loss per experiment here

        return self.run_sesop(sess)

    def run_sesop(self, sess):

        # Get batch for sesop:
        sess.run(self.models[0].get_batch_provider().set_batch_size_op(self.params.values()[0].sesop_batch_size))
        sesop_data, sesop_labels = sess.run(self.models[0].get_batch_provider().batch())
        sess.run(self.models[0].get_batch_provider().set_batch_size_op(self.params.values()[0].batch_size))

        #loss per experiment
        losses = {}

        for e in self.experiments:
            models = e.models
            master_model = models[0]
            worker_models = models[1:]

            assert (master_model.node_id == 0)

            for worker in worker_models:
                sess.run(worker.push_to_master_op())

            #Now optimize by alpha
            feed_dict = {}
            for m in e.models:
                inputs, labels = m.get_inputs()
                feed_dict[inputs] = sesop_data
                feed_dict[labels] = sesop_labels

            self.params[master_model].cg.minimize(sess, feed_dict=feed_dict)

            #Update the history:
            sess.run(master_model.hvar_mgr.all_history_update_ops())

            #Now send the results back to the workers
            for worker in worker_models:
                sess.run(worker.pull_from_master_op())

            losses[e] = sess.run([m.loss() for m in e.models], feed_dict=feed_dict)

        self.sesop_runs += 1
        self.curr_iter += 1

        return losses

