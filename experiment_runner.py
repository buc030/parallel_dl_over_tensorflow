

import tensorflow as tf
from seboost import SeboostOptimizer
from dataset_manager import DatasetManager
from batch_provider import BatchProvider
from model import Model
import experiments_manager
import progressbar
from experiment import Experiment
from batch_provider import SimpleBatchProvider, CifarBatchProvider
from cifar_input import build_input
import numpy as np
import tqdm

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
                if force_rerun == False and _e.get_number_of_ran_iterations() > 0:
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

        self.bs = experiments[0].getFlagValue('b')
        self.epochs = experiments[0].getFlagValue('epochs')
        self.dataset_size = experiments[0].getFlagValue('dataset_size')
        self.input_dim = experiments[0].getFlagValue('dim')
        self.output_dim = experiments[0].getFlagValue('output_dim')

        self.batch_size = experiments[0].getFlagValue('b')
        self.sesop_batch_size = experiments[0].getFlagValue('sesop_batch_size')
        self.careless_batch_size = 1000

        self.train_dataset_size, self.test_dataset_size = experiments[0].getDatasetSize()


    def dump_results(self):
        for e in self.experiments:
            experiments_manager.ExperimentsManager.get().dump_experiment(e)

    #return a list of the models, losses, accuracies, after the done experiments were removed
    def remove_finished_experiments(self):
        models, losses, accuracies, stages = [], [], [], []
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
        stages = [m.stage for m in models]
        #stages = []

        return models, losses, accuracies, stages

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
            with tf.device('/cpu:' + str(i%12)):
                if model == 'simple':
                    with tf.variable_scope('simple_batch_provider_' + str(i)) as scope:
                        self.batch_providers.append(
                            SimpleBatchProvider(input_dim=self.input_dim, output_dim=self.output_dim, \
                                                dataset_size=self.dataset_size, \
                                                batch_sizes=[self.bs, self.sesop_batch_size]))
                        scope.reuse_variables()

                elif model == 'mnist':
                    assert (False)
                elif model == 'cifar10':
                    with tf.variable_scope('cifar10_batch_provider_' + str(i)) as scope:
                        self.batch_providers.append(
                            CifarBatchProvider(batch_sizes=[self.bs, self.sesop_batch_size, self.careless_batch_size], \
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
        tf.get_default_graph()._unsafe_unfinalize()

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        #'grpc://' + tf_server
        # with tf.Session('grpc://' + tf_server, config=config) as sess:
        with tf.Session(target='grpc://localhost:2222', config=config) as sess:

            print 'building models (first e = '  + str(self.experiments[0]) + ')'
            i = 0
            expr_num = 0

            self.init_batch_providers(sess)


            for e in tqdm.tqdm(self.experiments):
            #for e in self.experiments:
                with tf.variable_scope('experiment_' + str(expr_num)):
                    e.init_models(i%4, self.batch_providers)
                i += 1
                expr_num += 1




            print 'Setting up optimizers'
            optimizer = SeboostOptimizer(self.experiments)

            print 'init vars'
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            print 'Reload/Dumping models'
            self.dump_results()
            for e in tqdm.tqdm(self.experiments):
                if e.get_number_of_ran_iterations() > 0 and self.force_rerun == False:
                    for m in e.models:
                        m.init_from_checkpoint(sess)
                else:
                    for m in e.models:
                        m.dump_checkpoint(sess)

            sess.graph.finalize()

            # we must start queue_runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for e in tqdm.tqdm(self.experiments):
                for m in e.models:
                    sess.run(m.stage)

            print 'Write graph into tensorboard'
            writer = tf.summary.FileWriter('/tmp/generated_data/' + '1')
            writer.add_graph(sess.graph)
            #self.epochs = 0
            # Progress bar:
            for epoch in range(self.epochs):

                print 'Setup losses'
                models, _, accuracies, stages = self.remove_finished_experiments()

                # run 20 steps (full batch optimization to start with)
                print 'epoch #' + str([e.get_number_of_ran_epochs() for e in self.experiments])
                print 'Computing train and test Accuracy'
                train_error, test_error = np.zeros(len(models)), np.zeros(len(models))


                #assert (self.train_dataset_size % self.careless_batch_size == 0)
                #assert (self.test_dataset_size % self.careless_batch_size == 0)
                for m in models:
                    m.batch_provider.set_source(sess, self.careless_batch_size, True)
                for i in tqdm.tqdm(range(self.train_dataset_size / self.careless_batch_size)):
                    train_error +=  np.array(sess.run(accuracies + stages)[:len(accuracies)])
                train_error /= float(self.train_dataset_size / self.careless_batch_size)
                print 'Train Accuracy = ' + str(train_error)

                for m in models:
                    m.batch_provider.set_source(sess, self.careless_batch_size, False)
                for i in tqdm.tqdm(range(self.test_dataset_size / self.careless_batch_size)):
                    test_error += np.array(sess.run(accuracies + stages)[:len(accuracies)])
                test_error /= float(self.test_dataset_size / self.careless_batch_size)
                print 'Test Accuracy = ' + str(test_error)

                print 'Dumping results....'
                self.add_experiemnts_results(train_error, test_error)
                self.dump_results()
                for m in tqdm.tqdm(models):
                    m.dump_checkpoint(sess)

                print 'Training'
                optimizer.run_epoch(sess=sess, stages=stages)

                #dump eperiments before continue to next round!

                # writer.flush()
            coord.request_stop()
            coord.join(threads)


import experiment


def find_cifar_baseline():
    experiments = {}
    #for lr in [0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.025/2]:
    for lr in [0.2, 0.1, 0.05, 0.025]:
    #for lr in [0.4, 0.5, 0.6, 0.7]:
        experiments[len(experiments)] = experiment.Experiment(
            {
                'model': 'cifar10',
                'b': 128,
                'lr': lr,
                'sesop_batch_size': 1000,
                'sesop_freq': (1.0 / 50000.0),  # sesop every 1 epochs (no sesop)
                'hSize': 0,
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

experiments = find_cifar_baseline()
#experiments = find_cifar_history()
runner = ExperimentRunner(experiments, force_rerun=True)
runner.run()
