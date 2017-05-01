

import tensorflow as tf
from tensorflow.python.training.queue_runner_impl import QueueRunner

from seboost import SeboostOptimizer
import experiments_manager
import progressbar
from experiment import Experiment
from batch_provider import SimpleBatchProvider, CifarBatchProvider
import numpy as np
import tqdm
import os

class ExperimentRunner:

    def assert_all_experiments_has_same_flag(self, flagname):
        if len(self.experiments) == 0:
            return

        val = self.experiments[0].getFlagValue(flagname)
        for e in self.experiments:
            assert (val == e.getFlagValue(flagname))

    def __init__(self, experiments, force_rerun = False):

        self.force_rerun = force_rerun
        experiments = list(experiments.values())
        self.experiments = []

        for e in experiments:
            _e = experiments_manager.ExperimentsManager.get().load_experiment(e)
            if _e is not None:
                if force_rerun == False and (_e.get_number_of_ran_iterations() > 0 or _e.get_number_of_ran_epochs() > 0):
                    if len(_e.results[0].trainError) >= _e.getFlagValue('epochs'):
                        print 'Experiment ' + str(_e) + ' already ran!'
                    else:
                        print 'Experiment ' + str(_e) + ' ran for ' + str(len(_e.results[0].trainErrorPerItereation)) + ' iterations...'
                        self.experiments.append(_e)
                    continue
            self.experiments.append(e)



        self.assert_all_experiments_has_same_flag('dataset_size')
        self.assert_all_experiments_has_same_flag('dim')
        self.assert_all_experiments_has_same_flag('epochs')
        self.assert_all_experiments_has_same_flag('b')
        self.assert_all_experiments_has_same_flag('sesop_freq')
        self.assert_all_experiments_has_same_flag('sesop_batch_size')
        self.assert_all_experiments_has_same_flag('model')
        self.assert_all_experiments_has_same_flag('hSize')
        self.assert_all_experiments_has_same_flag('nodes')

        self.epochs = experiments[0].getFlagValue('epochs')
        self.input_dim = experiments[0].getFlagValue('dim')
        self.output_dim = experiments[0].getFlagValue('output_dim')

        self.batch_size = experiments[0].getFlagValue('b')

        self.train_dataset_size, self.test_dataset_size = experiments[0].getDatasetSize()


    def dump_results(self):
        for e in self.experiments:
            experiments_manager.ExperimentsManager.get().dump_experiment(e)

    #return a list of the models, losses, accuracies, after the done experiments were removed
    def remove_finished_experiments(self):
        models, losses, accuracies = [], [], []
        to_remove = []
        for e in self.experiments:
            if len(e.results[0].trainErrorPerItereation) >= e.getFlagValue('epochs')*(self.train_dataset_size/self.batch_size):
                print 'Experiment ' + str(e) + ' is done! Removing it from run...'
                to_remove.append(e)

        for e in to_remove:
            self.experiments.remove(e)

        for e in self.experiments:
            models.extend(e.models)

        losses = [m.loss() for m in models]
        accuracies = [m.accuracy() for m in models]


        return models, losses, accuracies

    def getSharedFlagValue(self, flagname):
        res = self.experiments[0].getFlagValue(flagname)
        self.assert_all_experiments_has_same_flag(flagname)
        return res

    def init_batch_providers(self, sess):
        self.batch_providers = []

        model = self.getSharedFlagValue('model')
        max_nodes = max([e.getFlagValue('nodes') for e in self.experiments])

        #Every node needs to have a different batch provider to make sure they see diff data
        for i in range(max_nodes):
            with tf.device('/cpu:*'):
                if model == 'simple':
                    with tf.name_scope('simple_batch_provider_' + str(i)) as scope:
                        self.batch_providers.append(
                            SimpleBatchProvider(input_dim=self.input_dim, output_dim=self.output_dim, \
                                                dataset_size=self.train_dataset_size, \
                                                batch_size=self.batch_size))

                elif model == 'mnist':
                    assert (False)
                elif model == 'cifar10':
                    with tf.variable_scope('cifar10_batch_provider_' + str(i)) as scope:
                        self.batch_providers.append(
                            CifarBatchProvider(batch_sizes=[self.batch_size], \
                                               train_threads=4*len(self.experiments)))

        return self.batch_providers

    def add_experiemnts_results(self, train_error, test_error):
        i = 0
        for e in self.experiments:
            for model_idx in range(len(e.models)):
                e.add_train_error(model_idx, train_error[i])
                e.add_test_error(model_idx, test_error[i])
                i += 1


    #run all the experiments in parallel
    def run(self):
        if len(self.experiments) == 0:
            print 'Nothing to run!'
            return

        tf.reset_default_graph()

        #set numpy seed:
        np.random.seed(6352)


        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        #'grpc://' + tf_server
        with tf.Session(config=config) as sess:
        #with tf.Session(target='grpc://localhost:2222', config=config) as sess:

            # Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
            # from tensorflow.python import debug as tf_debug
            #
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            pid = os.getpid()
            print 'building models (first e = '  + str(self.experiments[0]) + ')'
            gpu = 0
            expr_num = 0

            self.init_batch_providers(sess)

            for e in tqdm.tqdm(self.experiments):
                with tf.variable_scope('pid_' + str(pid) + '_experiment_' + str(expr_num)):
                    gpu += e.init_models(gpu, self.batch_providers)
                expr_num += 1


            print 'Setting up optimizers'
            with tf.name_scope('seboost'):
                optimizer = SeboostOptimizer(self.experiments)
            # with tf.name_scope('stochastic_cg'):
            #     optimizer = StochasticCGOptimizer(self.experiments[0].models[0].loss(), self.experiments[0].models[0].hvar_mgr.all_trainable_weights(), \
            #                                       self.experiments[0].models[0].get_extra_train_ops(), 10)
            #     print sess
            #     optimizer.init_ops(sess)

            print 'Write graph into tensorboard'
            writer = tf.summary.FileWriter('/tmp/generated_data/' + '1')
            writer.add_graph(sess.graph)

            merged = tf.summary.merge_all()
            optimizer.writer = writer
            optimizer.merged = merged

            print 'init vars'
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            print 'Reload/Dumping models'
            self.dump_results()
            for e in self.experiments:
                if e.get_number_of_ran_iterations() > 0 and self.force_rerun == False:
                    for m in e.models:
                        print 'Reloading model...'
                        m.init_from_checkpoint(sess)
                else:
                    for m in e.models:
                        print 'Dumping model...'
                        m.dump_checkpoint(sess)

            sess.graph.finalize()

            # we must start queue_runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for batch_provider in self.batch_providers:
                batch_provider.custom_runner.start_threads(sess, n_train_threads=1, n_test_threads=1)

            print 'pulling initial weights from master'
            for e in self.experiments:
                for worker in e.models[1:]:
                    sess.run(worker.pull_from_master_op())

            models, _, accuracies = self.remove_finished_experiments()

            for epoch in range(self.epochs):

                print 'Setup losses'
                models, _, accuracies = self.remove_finished_experiments()

                # run 20 steps (full batch optimization to start with)

                # master_weights = models[0].hvar_mgr.all_trainable_weights()
                # worker_weights = models[1].hvar_mgr.all_trainable_weights()
                # print 'master_weights = ' + str(sess.run(master_weights[0]))
                # print 'worker_weights = ' + str(sess.run(worker_weights[0]))

                print 'Computing train and test Accuracy'
                train_error, test_error = np.zeros(len(models)), np.zeros(len(models))

                for m in models:
                    m.batch_provider.set_data_source(sess, 'train')
                for i in tqdm.tqdm(range((self.train_dataset_size/1) / self.batch_size)):
                    train_error +=  np.array(sess.run(accuracies))
                train_error /= float((self.train_dataset_size/1) / self.batch_size)
                print 'Train Accuracy = ' + str(train_error)

                for m in models:
                    m.batch_provider.set_data_source(sess, 'test')
                for i in tqdm.tqdm(range(self.test_dataset_size / self.batch_size)):

                    #steps = {'accuracies' : accuracies, 'stages' : stages, 'merged' : merged}
                    steps = {'accuracies': accuracies}
                    steps = sess.run(steps)

                    # writer.add_summary(steps['merged'], epoch*(self.test_dataset_size / self.careless_batch_size) + i)
                    # writer.flush()

                    test_error += np.array(steps['accuracies'])
                test_error /= float(self.test_dataset_size / self.batch_size)
                print 'Test Accuracy = ' + str(test_error)

                print 'Dumping results....'
                self.add_experiemnts_results(train_error, test_error)
                self.dump_results()
                for m in tqdm.tqdm(models):
                    m.dump_checkpoint(sess)

                print 'Start training Epoch #' + str([e.get_number_of_ran_epochs() for e in self.experiments])
                print 'Training'

                # for m in models:
                #     m.batch_provider.set_source(sess, self.batch_size, 1)
                # for i in tqdm.tqdm(range(self.train_dataset_size/ self.batch_size)):
                #     optimizer.run_iter(sess)
                #     for e in self.experiments:
                #         for model_idx in range(len(e.models)):
                #             e.add_iteration_train_error(model_idx, optimizer.losses[-1])

                for m in models:
                    m.batch_provider.set_data_source(sess, 'train')
                optimizer.run_epoch(sess=sess)

                #dump eperiments before continue to next round!

                # writer.flush()

            print '########### request_stop ###########'
            coord.request_stop()
            print '########### join ###########'
            coord.join(threads)
            print '########### after join ###########'


import experiment


def find_cifar_baseline():
    experiments = {}
    #for lr in [0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.025/2]:
    #for lr in [0.2, 0.1, 0.05, 0.025]:
    for lr in [0.2, 0.1, 0.05, 0.025]:
    #for lr in [0.3, 0.4, 0.5, 0.6]:
    #for lr in [0.7, 0.8, 0.9, 1.0]:
    #for lr in [1.1, 1.2, 1.3, 1.4]:
    #for lr in [0.8]:
        experiments[len(experiments)] = experiment.Experiment(
        {
                'model': 'cifar10',
                'b': 128,
                'lr': lr,
                'sesop_batch_size': 1000,
                'sesop_batch_mult': 1,
                'sesop_freq': 2e-05,  # sesop every 1 epochs (no sesop)
                'hSize': 0,
                'epochs': 100,
                # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
                'nodes': 1,
                'num_residual_units': 4,
                # Not relevant!
                'dim': None,
                'output_dim': None,
                'dataset_size': None,
                'hidden_layers_num': None,
                'hidden_layers_size': None

        })
    return experiments

def find_cifar_multinode(n):
    experiments = {}
    #for lr in [0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.025/2]:
    #for lr in [0.2, 0.1, 0.05, 0.025]:
    #0.101 CG
    #0.1 NG
    for lr in [0.101]:
    #for lr in [0.3, 0.4, 0.5, 0.6]:
    #for lr in [0.7, 0.8, 0.9, 1.0]:
    #for lr in [1.1, 1.2, 1.3, 1.4]:
    #for lr in [0.8]:
        experiments[len(experiments)] = experiment.Experiment(
        {
                'model': 'cifar10',
                'b': 128,
                'lr': lr,
                'sesop_batch_size' : 0,
                'sesop_batch_mult': 1,
                'sesop_freq': 1.0/390.0, #(1.0 / 391.0),  # sesop every 1 epochs (no sesop)
            # SV DEBUG
                'hSize': 0,
                'epochs': 100,
                # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
                'nodes': n,
            #SV DEBUG
                'num_residual_units': 4,
                # Not relevant!
                'dim': None,
                'output_dim': None,
                'dataset_size': None,
                'hidden_layers_num': None,
                'hidden_layers_size': None

        })
    return experiments


def find_cifar_history():
    experiments = {}
    hs =  [1  ,    2   ,      4    ,   8   ,   16  ,     32    ,   64]

    for h in hs:
        experiments[len(experiments)] = experiment.Experiment(
            {
                'model': 'cifar10',
                'b': 128,
                'lr': 0.1,
                'sesop_batch_size': 1000,
                'sesop_freq': (1.0 / 50000.0),  # sesop every 1 epochs (no sesop)
                'hSize': h,
                'epochs': 250,
                # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
                'nodes': 1,

                # Not relevant!
                'dim': None,
                'output_dim': None,
                'dataset_size': None,
                'hidden_layers_num': None,
                'hidden_layers_size': None

            })

    return experiments


def run_cifar_expr():
    experiments = {}
    ns =  [1    ,   1  ,    ]#2   ,   4       ,4      ]
    hs =  [0    ,   0  ,    ]#1   ,   1       ,4      ]
    lrs = [0.1  ,   0.05,    ]#0.05,   0.025   ,0.025  ]

    for n,h,lr in zip(ns, hs, lrs):
        experiments[len(experiments)] = experiment.Experiment(
            {
                'model': 'cifar10',
                'b': 128,
                'lr': lr,
                'sesop_batch_size': 1000,
                'sesop_freq': (1.0 / 50000.0),  # sesop every 1 epochs (no sesop)
                'hSize': h,
                'epochs': 250,
            # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
                'nodes': n,

                # Not relevant!
                'dim': None,
                'output_dim': None,
                'dataset_size': None,
                'hidden_layers_num': None,
                'hidden_layers_size': None

            })

    return experiments

def simple():
    experiments = {}
    for lr in [0.1]:
        experiments[len(experiments)] = experiment.Experiment(
        {
            'model': 'simple',
            'b': 100,
            'lr': lr,
            'sesop_batch_size': 0,
            'sesop_batch_mult': 1,
            'sesop_freq': 1.0 / 50.0,  # (1.0 / 391.0),  # sesop every 1 epochs (no sesop)
            'hSize': 0,
            'nodes': 1,
            'dim': 10,
            'output_dim': 1,
            'dataset_size': 5000,
            'hidden_layers_num': 3,
            'hidden_layers_size': 10,


            'epochs': 30,
            'num_residual_units': None


        })
    return experiments


def find_simple_baseline():
    experiments = {}

    #0.08 is the winner
    for lr in [0.1, 0.09, 0.08, 0.07, 0.06, 0.05]:
        experiments[len(experiments)] = experiment.Experiment(
        {
            'model': 'simple',
            'b': 100,
            'lr': lr,
            'sesop_batch_size': 0,
            'sesop_batch_mult': 1,
            'sesop_freq': 1.0 / 50.0,  # (1.0 / 391.0),  # sesop every 1 epochs (no sesop)
            'hSize': 0,
            'nodes': 1,
            'dim': 10,
            'output_dim': 1,
            'dataset_size': 50000,
            'hidden_layers_num': 3,
            'hidden_layers_size': 100,


            'epochs': 30,
            'num_residual_units': None


        })
    return experiments

def simple_with_history_baseline(h, sesop_batch_mult):
    experiments = {}
    experiments[len(experiments)] = experiment.Experiment(
        {
            'model': 'simple',
            'b': 100,
            'lr': 0.08,
            'sesop_batch_size': 0,
            'sesop_batch_mult': sesop_batch_mult,
            'sesop_freq': 1.0 / 500.0,  # (1.0 / 391.0),  # sesop every 1 epochs (no sesop)
            'hSize': h,
            'nodes': 1,
            'dim': 10,
            'output_dim': 1,
            'dataset_size': 50000,
            'hidden_layers_num': 3,
            'hidden_layers_size': 100,


            'epochs': 30,
            'num_residual_units': None


    })
    return experiments


def simple_multinode(n, h, sesop_batch_mult):
    experiments = {}
    experiments[len(experiments)] = experiment.Experiment(
        {
            'model': 'simple',
            'b': 100,
            'lr': 0.08,
            'sesop_batch_size': 0,
            'sesop_batch_mult': sesop_batch_mult,
            'sesop_freq': 1.0 / 50.0,  # (1.0 / 391.0),  # sesop every 1 epochs (no sesop)
            'hSize': h,
            'nodes': n,
            'dim': 10,
            'output_dim': 1,
            'dataset_size': 5000,
            'hidden_layers_num': 3,
            'hidden_layers_size': 100,


            'epochs': 30,
            'num_residual_units': None


    })
    return experiments

