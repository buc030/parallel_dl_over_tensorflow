

import tensorflow as tf
import numpy as np
#import argparse
import sys
import threading
from threading import current_thread

import experiments_manager
from summary_manager import SummaryManager
from tf_utils import *
from natural_gradient import NaturalGradientOptimizer

import tqdm

class SeboostOptimizerParams:
    def __init__(self, model):

        self.lr = model.experiment.getFlagValue('lr')
        self.sgd_steps = int(1.0/model.experiment.getFlagValue('sesop_freq'))
        self.sesop_batch_size = model.experiment.sesop_batch_size
        self.batch_size = model.experiment.getFlagValue('b')
        self.sesop_batch_mult = model.experiment.sesop_batch_mult

        self.cg_var_list = model.hvar_mgr.all_trainable_alphas()
        if len(self.cg_var_list) > 0:
            self.cg = tf.contrib.opt.ScipyOptimizerInterface(loss=model.loss(), var_list=self.cg_var_list, iteration_mult=self.sesop_batch_mult,\
                                                             method='BFGS', options={'maxiter': 100, 'gtol' : 1e-5})

        # if len(self.cg_var_list) > 0:
        #     self.cg = NaturalGradientOptimizer(model.loss(), model.model.predictions, self.cg_var_list)

#TODO: try big subspace with strong optimization

#This runs the same optimization process for 'models' with corosponding metaparameters defined in 'experiments'
#All the models see the exact same data!
class SeboostOptimizer:

    #batched_input, batched_labels are tensors that prodece batches
    #is_training is a tensor that will be true while training and false while testing
    #we run CG once in sesop_freq iterations
    def __init__(self, experiments):
        self.sesop_runs = 0
        self.train_dataset_size, self.test_dataset_size = experiments[0].getDatasetSize()

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

    def run_epoch(self, sess):

        for m in self.models:
            m.batch_provider.set_data_source(sess, 'train')

        for i in tqdm.tqdm(range(self.train_dataset_size/self.batch_size)):
            self.run_iter(sess)

        print 'Actual sesop freq is: ' + str(float(self.sesop_runs + 1) / (self.curr_iter + 1))


    def run_simple_iter(self, sess):
        #print '################### RUNNING SIMPLE ITER ################'
        _, losses = sess.run([self.train_steps, self.losses])

        i = 0
        for e in self.experiments:
            for model_idx in range(len(e.models)):
                e.add_iteration_train_error(model_idx, losses[i])
                i += 1

        self.curr_iter += 1
        return None  # TODO: need to return loss per experiment here

    def run_iter(self, sess):
        if self.curr_iter % self.params.values()[0].sgd_steps != 0:
            return self.run_simple_iter(sess)

        return self.run_sesop(sess)

    def dump_debug(self, sess, master_model, feed_dict, suffix):
        return
        with open('debug_' + suffix, 'w') as f:
            f.write('loss = ' + str(sess.run(master_model.loss(), feed_dict=feed_dict)) + '\n')

            #
            # f.write('_images = ' + str(sess.run(master_model.model._images, feed_dict=feed_dict)) + '\n')
            # f.write('res_layers[0] = ' + str(sess.run(master_model.model.res_layers[0], feed_dict=feed_dict)) + '\n')
            # f.write('res_layers[-1] = ' + str(sess.run(master_model.model.res_layers[-1], feed_dict=feed_dict)) + '\n')
            #
            # f.write('predictions = ' + str(sess.run(master_model.model.predictions, feed_dict=feed_dict)) + '\n')
            #
            # f.write('cost = ' + str(sess.run(master_model.model.cost, feed_dict=feed_dict)) + '\n')
            # f.write('cost_before_decay = ' + str(sess.run(master_model.model.cost_before_decay, feed_dict=feed_dict)) + '\n')

            #cost_before_decay
            for debug_hvar in master_model.hvar_mgr.all_hvars:
                f.write('debug_hvar.out() = ' + str(sess.run(debug_hvar.out())) + '\n')
                #f.write('debug_hvar.var = ' + str(sess.run(debug_hvar.var)))
                f.write('---------------------')

    def run_sesop(self, sess):
        if self.experiments[0].getFlagValue('hSize') == 0 and self.experiments[0].getFlagValue('nodes') == 1:
            self.run_simple_iter(sess)
            self.curr_iter += 1
            return None


        #loss per experiment
        losses = {}

        for e in self.experiments:

            master_model = e.models[0]
            worker_models = e.models[1:]

            assert (master_model.node_id == 0)

            for worker in worker_models:
                sess.run(worker.push_to_master_op())

            b4_sesop, after_sesop = master_model.hvar_mgr.all_history_update_ops()
            zero_alpha = master_model.hvar_mgr.all_zero_alpha_ops()

            #Push the new direction into P:
            sess.run(b4_sesop)

            #Now optimize by alpha
            master_model.batch_provider.set_data_source(sess, 'train')

            feed_dicts = []
            for i in range(master_model.experiment.sesop_batch_mult):
                feed_dict = master_model.get_shared_feed(sess, worker_models)
                feed_dicts.append(feed_dict)

            losses[e] = sess.run([m.loss() for m in e.models], feed_dict=feed_dicts[0])

            loss_b4_sesop = master_model.calc_train_accuracy(sess, batch_size=self.batch_size, train_dataset_size=self.train_dataset_size)

            self.dump_debug(sess, master_model, feed_dicts[0], 'loss_before_sesop')

            e.start_new_subspace_optimization()
            #loss_callback=debug_loss_callback
            def debug_loss_callback(loss, grad):
                e.add_loss_during_supspace_optimization(loss)
                e.add_grad_norm_during_supspace_optimization(np.linalg.norm(grad, 2))

            print 'size of subspace = ' + str(len(self.params[master_model].cg_var_list))
            self.params[master_model].cg.minimize(sess, feed_dicts=feed_dicts, loss_callback=debug_loss_callback)

            loss_after_sesop = master_model.calc_train_accuracy(sess, batch_size=self.batch_size,
                                                             train_dataset_size=self.train_dataset_size)

            e.add_debug_sesop(0, loss_b4_sesop, loss_after_sesop)

            sess.run(after_sesop) #after this 'snapshot' is updated with the result of the sesop

            sess.run(zero_alpha)

            #Now send the results back to the workers
            for worker in worker_models:
                #TODO: worker.assert_have_no_alpa_or_history_or_replicas()
                sess.run(worker.pull_from_master_op()) #assign(var, snapshot)

            e.add_debug_sesop_on_sesop_batch(0, losses[e], sess.run(master_model.loss(), feed_dict=feed_dicts[0]))

            losses[e] = sess.run([m.loss() for m in e.models], feed_dict=feed_dicts[0])

            self.dump_debug(sess, master_model, feed_dicts[0], 'master')
            if len(worker_models) > 0:
                self.dump_debug(sess, worker_models[0], feed_dicts[0], 'worker')

            # master_weights = master_model.hvar_mgr.all_trainable_weights()
            # worker_weights = worker_models[0].hvar_mgr.all_trainable_weights()
            # print 'master_weights = ' + str(sess.run(master_weights[0]))
            # print 'worker_weights = ' + str(sess.run(worker_weights[0]))

            master_model.batch_provider.set_data_source(sess, 'train')

        self.sesop_runs += 1
        self.curr_iter += 1

        return None

