

import tensorflow as tf
import numpy as np
#import argparse
import sys
import threading
from threading import current_thread

import experiments_manager
from summary_manager import SummaryManager
from tf_utils import *

import tqdm

class SeboostOptimizerParams:
    def __init__(self, model):

        self.lr = model.experiment.getFlagValue('lr')
        self.sgd_steps = int(1.0/model.experiment.getFlagValue('sesop_freq'))
        self.sesop_batch_size = model.experiment.sesop_batch_size
        self.batch_size = model.experiment.getFlagValue('b')

        self.cg_var_list = model.hvar_mgr.all_trainable_alphas()
        if len(self.cg_var_list) > 0:
            self.cg = tf.contrib.opt.ScipyOptimizerInterface(model.loss(), var_list=self.cg_var_list, \
                                                         method='CG', options={'maxiter': 10})



#This runs the same optimization process for 'models' with corosponding metaparameters defined in 'experiments'
#All the models see the exact same data!
class SeboostOptimizer:

    #batched_input, batched_labels are tensors that prodece batches
    #is_training is a tensor that will be true while training and false while testing
    #we run CG once in sesop_freq iterations
    def __init__(self, experiments):
        self.sesop_runs = 0
        self.dataset_size, _ = experiments[0].getDatasetSize()

        self.batch_size = experiments[0].bs
        self.models = []
        self.experiments = experiments

        for e in experiments:
            for m in e.models:
                self.models.append(m)

        self.params = {}
        for m in tqdm.tqdm(self.models):
            self.params[m] = SeboostOptimizerParams(m)


        self.losses = [m.loss() for m in self.models]
        #self.train_steps = [p.train_step for p in self.params.values()]
        self.train_steps = []
        for m in self.models:
            self.train_steps.extend(m.train_op())

        print 'self.train_steps = ' + str(self.train_steps)
        self.curr_iter = 1

    def run_epoch(self, sess, stages):

        for m in self.models:
            m.batch_provider.set_source(sess, self.batch_size, True)


        for i in tqdm.tqdm(range(self.dataset_size/self.batch_size)):
            self.run_iter(sess, stages)


        print 'Actual sesop freq is: ' + str(float(self.sesop_runs + 1) / (self.curr_iter + 1))


    def run_iter(self, sess, stages):
        if self.curr_iter % self.params.values()[0].sgd_steps != 0:# or len(self.params.values()[0].cg_var_list) == 0:
            _, losses, __ = sess.run([self.train_steps, self.losses, stages])

            i = 0
            for e in self.experiments:
                for model_idx in range(len(e.models)):
                    e.add_iteration_train_error(model_idx, losses[i])
                    i += 1

            self.curr_iter += 1
            return None  # TODO: need to return loss per experiment here

        return self.run_sesop(sess)

    def run_sesop(self, sess):


        #loss per experiment
        losses = {}

        for e in self.experiments:
            if e.getFlagValue('hSize') == 0 and e.getFlagValue('nodes') == 1:
                sess.run([self.params[m].train_step for m in e.models])
                losses[e] = sess.run([m.loss() for m in e.models])
                continue

            master_model = e.models[0]
            worker_models = e.models[1:]

            assert (master_model.node_id == 0)

            for worker in worker_models:
                sess.run(worker.push_to_master_op())

            b4_sesop, after_sesop = master_model.hvar_mgr.all_history_update_ops()
            #Push the new direction into P:
            sess.run(b4_sesop)

            #Now optimize by alpha

            #loss_b4_sesop = sess.run(master_model.loss() ,feed_dict={inputs: full_data, labels: full_labels})
            feed_dict = master_model.get_shared_feed(sess, self.params[master_model].sesop_batch_size, True)
            assert (len(feed_dict) == 2 * len(e.models))

            self.params[master_model].cg.minimize(sess, feed_dict=feed_dict)
            sess.run(after_sesop)

            #loss_after_sesop = sess.run(master_model.loss(), feed_dict={inputs: full_data, labels: full_labels})


            #print 'loss_b4_sesop - loss_after_sesop = ' + str(loss_b4_sesop - loss_after_sesop)
            #assert(False)
            #master_model.log_loss_b4_minus_after(sess, loss_b4_sesop - loss_after_sesop)


            #master_model.dump_to_tensorboard(sess)
            #master_model.summary_mgr.writer.flush()
            #Update the history:


            #Now send the results back to the workers
            for worker in worker_models:
                #TODO: worker.assert_have_no_alpa_or_history_or_replicas()
                sess.run(worker.pull_from_master_op())


            losses[e] = sess.run([m.loss() for m in e.models], feed_dict=feed_dict)


        self.sesop_runs += 1
        self.curr_iter += 1

        return losses

