


import sys

from tf_utils import *
from natural_gradient import NaturalGradientOptimizer

from my_external_optimizer import ScipyOptimizerInterface

import tqdm
import debug_utils
import Queue


class SeboostOptimizerParams:
    def get_optimizer(self, model, cg_var_list):
        opt = model.experiment.getFlagValue('subspace_optimizer')

        self.grad_norm = tf.global_norm(tf.gradients(model.loss(), cg_var_list))
        self.grad_norm_placeholder = tf.placeholder(dtype=tf.float32)
        if opt == 'newton':
            import newton_optimizer
            return newton_optimizer.NewtonOptimizer(model.loss()/self.grad_norm_placeholder, cg_var_list)
        elif opt == 'CG':
            return ScipyOptimizerInterface(loss=model.loss()/self.grad_norm_placeholder, var_list=cg_var_list, \
                                           method='CG', options={'maxiter': 200 * len(cg_var_list), 'gtol': 1e-6})
        elif opt == 'BFGS':
            return ScipyOptimizerInterface(loss=model.loss()/self.grad_norm_placeholder, var_list=cg_var_list, \
                                           method='BFGS', options={'maxiter': 200 * len(cg_var_list), 'gtol': 1e-6})
        assert (False)

    def __init__(self, model):

        self.lr = model.experiment.getFlagValue('lr')
        self.sgd_steps = int(1.0/model.experiment.getFlagValue('sesop_freq'))
        self.sesop_batch_size = model.experiment.sesop_batch_size
        self.batch_size = model.experiment.getFlagValue('b')
        self.sesop_batch_mult = model.experiment.sesop_batch_mult

        self.cg_var_list = model.hvar_mgr.all_trainable_alphas()
        if len(self.cg_var_list) > 0:
            with tf.device('/gpu:' + str(model.gpu)):
                self.cg = self.get_optimizer(model, self.cg_var_list)


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
        self.grad_norms = [m.grad_norm for m in self.models]
        self.weights_norms = [m.weights_norm for m in self.models]
        self.input_norms = [m.input_norm for m in self.models]

        #self.train_steps = [p.train_step for p in self.params.values()]
        self.train_steps = []
        for m in self.models:
            self.train_steps.extend(m.train_op())

        print 'self.train_steps = ' + str(self.train_steps)
        self.curr_iter = 1

        self.loss_on_prev_sesop = None

        self.sesop_threads_que = Queue.Queue(len(experiments))
        for e in self.experiments:
            for model_idx in range(len(e.models)):
                e.push_sgd_epoch(model_idx)

    def run_epoch(self, sess):

        for m in self.models:
            m.batch_provider.set_data_source(sess, 'train')

        for i in tqdm.tqdm(range(self.train_dataset_size/self.batch_size)):
            self.run_iter(sess)


        res = self.run_sesop()
        print 'Actual sesop freq is: ' + str(float(self.sesop_runs + 1) / (self.curr_iter + 1))

    def run_simple_iter(self, sess):
        #print '############### RUNNING SIMPLE ITER ################'
        _, losses, grad_norms, weights_norms, input_norms = sess.run([self.train_steps, self.losses, self.grad_norms, self.weights_norms, self.input_norms])

        i = 0
        for e in self.experiments:
            for model_idx in range(len(e.models)):
                e.add_iteration_train_error(model_idx, losses[i])
                e.add_sgd_iter_grad_norm(model_idx, grad_norms[i])
                e.add_sgd_iter_weight_norm(model_idx, weights_norms[i])
                e.add_sgd_iter_input_norm(model_idx, input_norms[i])
                i += 1

        self.curr_iter += 1
        return None  # TODO: need to return loss per experiment here

    def run_iter(self, sess):
        if self.curr_iter % self.params.values()[0].sgd_steps != 0:
            return self.run_simple_iter(sess)

        for e in self.experiments:
            for model_idx in range(len(e.models)):
                e.push_sgd_epoch(model_idx)

        # loss_before_sesop = self.models[0].calc_train_accuracy(sess, batch_size=self.batch_size,
        #                                           train_dataset_size=self.train_dataset_size)
        #
        # if self.loss_on_prev_sesop is not None and loss_before_sesop > self.loss_on_prev_sesop:
        #     print '#### DIVIDING LEARNING RATE #####'
        #     for m in self.models:
        #         m.div_learning_rate(sess, 10)

        res = self.run_sesop()

        # self.loss_on_prev_sesop = self.models[0].calc_train_accuracy(sess, batch_size=self.batch_size,
        #                                       train_dataset_size=self.train_dataset_size)

        return res

    def dump_debug(self, sess, master_model, feed_dict, suffix):
        if debug_utils.DEBUG_LEVEL < 2:
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

    def run_sesop_on_experiment(self, sess, e):
        if e.getFlagValue('hSize') == 0 and e.getFlagValue('nodes') == 1:
            self.run_simple_iter(sess)
            return

        master_model = e.models[0]
        worker_models = e.models[1:]

        assert (master_model.node_id == 0)

        sess.run([worker.push_to_master_op() for worker in worker_models])

        b4_sesop, after_sesop = master_model.hvar_mgr.all_history_update_ops()
        zero_alpha = master_model.hvar_mgr.all_zero_alpha_ops()

        sess.run(master_model.hvar_mgr.assert_alphas_are_zero())
        # Push the new direction into P, and set initial alpha values for replicas
        sess.run(b4_sesop)

        # Now optimize by alpha
        master_model.batch_provider.set_data_source(sess, 'sesop')
        master_model.batch_provider.custom_runner.set_deque_batch_size(sess, self.batch_size * e.getFlagValue(
            'sesop_batch_mult'))
        feed_dicts = [master_model.get_shared_feed(sess, worker_models) for i in
                      range(1)]
        master_model.batch_provider.custom_runner.set_deque_batch_size(sess, self.batch_size)

        if debug_utils.DEBUG_LEVEL > 0:
            loss_b4_sesop_on_batch = avarge_on_feed_dicts(sess=sess, additional_feed_dict={},
                                                          target_ops=[master_model.loss()], feed_dicts=feed_dicts)
            loss_b4_sesop = master_model.calc_train_accuracy(sess, batch_size=self.batch_size,
                                                             train_dataset_size=self.train_dataset_size)

        e.start_new_subspace_optimization()

        # loss_callback=debug_loss_callback
        def debug_loss_callback(loss, grad):
            e.add_loss_during_supspace_optimization(loss)
            #SV DEBUG
            e.add_grad_norm_during_supspace_optimization(np.linalg.norm(grad, 2))
            # grad_norm = avarge_on_feed_dicts(sess=sess, additional_feed_dict={},
            #                                  target_ops=[self.params[master_model].grad_norm], feed_dicts=feed_dicts)
            # e.add_grad_norm_during_supspace_optimization(grad_norm)


        #######################################################################
        #######################################################################
        ########################## MINIMIZE STAGE #############################
        print 'size of subspace = ' + str(len(self.params[master_model].cg_var_list))

        grad_norm = 1.0
        if e.getFlagValue('NORMALIZE_DIRECTIONS') == True:
            sess.run(master_model.hvar_mgr.normalize_directions_ops())

        print 'grad_norm before ' + str(avarge_on_feed_dicts(sess=sess, additional_feed_dict={},
                                             target_ops=[self.params[master_model].grad_norm], feed_dicts=feed_dicts))

        if e.getFlagValue('NORMALIZE_DIRECTIONS') == True:
            grad_norm = avarge_on_feed_dicts(sess=sess, additional_feed_dict={},
                                             target_ops=[self.params[master_model].grad_norm], feed_dicts=feed_dicts)

            #SV INTERSTING
            #at a certain point, it is possible for the model to become really good on the sesop set, so the gradient becomes 0.0, or very close
            #at this point, we dont want to run subspace optimization.

            if grad_norm[0] < 1e-6:
                print 'BOOM not running sesop!'
            else:
                self.params[master_model].cg.minimize(sess, feed_dicts=feed_dicts, loss_callback=debug_loss_callback,
                                                      additional_feed_dict={
                                                          self.params[master_model].grad_norm_placeholder: grad_norm})
        else:
            self.params[master_model].cg.minimize(sess, feed_dicts=feed_dicts, loss_callback=debug_loss_callback,
                                              additional_feed_dict={
                                                  self.params[master_model].grad_norm_placeholder: grad_norm})

        if e.getFlagValue('NORMALIZE_DIRECTIONS') == True:
            grad_norm = avarge_on_feed_dicts(sess=sess, additional_feed_dict={},
                                             target_ops=[self.params[master_model].grad_norm], feed_dicts=feed_dicts)

        print 'grad_norm after ' + str(avarge_on_feed_dicts(sess=sess, additional_feed_dict={},
                                                                 target_ops=[self.params[master_model].grad_norm],
                                                                 feed_dicts=feed_dicts))
        ############### DONE MINIMIZING, RESULTS NOW ARE IN ALPHA ##############
        ########################################################################
        ########################################################################

        if debug_utils.DEBUG_LEVEL > 0:
            loss_after_sesop = master_model.calc_train_accuracy(sess, batch_size=self.batch_size,
                                                                train_dataset_size=self.train_dataset_size)
            e.add_debug_sesop(0, loss_b4_sesop, loss_after_sesop)
            e.add_debug_sesop_on_sesop_batch(0, loss_b4_sesop_on_batch,
                                             sess.run(master_model.loss(), feed_dict=feed_dicts[0]))

        sess.run(after_sesop)  # after this 'snapshot' is updated with the result of the sesop
        sess.run(zero_alpha)
        # Now send the results back to the workers
        for worker in worker_models:
            sess.run(worker.pull_from_master_op())  # assign(var, snapshot)

        self.dump_debug(sess, master_model, feed_dicts[0], 'master')
        if len(worker_models) > 0:
            self.dump_debug(sess, worker_models[0], feed_dicts[0], 'worker')

        # master_weights = master_model.hvar_mgr.all_trainable_weights()
        # worker_weights = worker_models[0].hvar_mgr.all_trainable_weights()
        # print 'master_weights = ' + str(sess.run(master_weights[0]))
        # print 'worker_weights = ' + str(sess.run(worker_weights[0]))

        master_model.batch_provider.set_data_source(sess, 'train')

        sess.run(master_model.hvar_mgr.assert_alphas_are_zero())

    def sesop_thread(self, sess):
        while True:
            e = self.sesop_threads_que.get(True, None)
            #print 'Running sesop'
            self.run_sesop_on_experiment(sess, e)
            #print 'Done Running sesop'
            self.sesop_threads_que.task_done()

    def start_threads(self, sess):
        threads = []
        for e in self.experiments:
            t = threading.Thread(target=self.sesop_thread, args=(sess, ))
            t.daemon = True
            t.start()
            threads.append(t)



    def run_sesop(self):
        # if len(self.experiments) == 1:
        #     res = self.run_sesop_on_experiment(sess, self.experiments[0])
        #     self.sesop_runs += 1
        #     self.curr_iter += 1
        #     return res
        #print 'Pushing sesop jobs'
        for e in self.experiments:
            self.sesop_threads_que.put(e)

        self.sesop_threads_que.join()
        #print 'Join sesop threads done'

        self.sesop_runs += 1
        self.curr_iter += 1

        return None

