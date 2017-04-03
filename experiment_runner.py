

import tensorflow as tf
from seboost import SeboostOptimizer
from dataset_manager import DatasetManager
from batch_provider import BatchProvider
from model import Model
import experiments_manager
import progressbar
from experiment import Experiment
from batch_provider import SimpleBatchProvider
from cifar_input import build_input
import numpy as np

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
                if force_rerun == False and len(_e.results) > 0:
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

        if experiments[0].getFlagValue('model') == 'simple':
            self.train_dataset_size = 5000
            self.test_dataset_size = 5000
        elif experiments[0].getFlagValue('model') == 'cifar10':
            self.train_dataset_size = 50000
            self.test_dataset_size = 10000
        else:
            assert(False)

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
        with tf.Session(config=config) as sess:

            print 'building models (first e = '  + str(self.experiments[0]) + ')'
            bar = progressbar.ProgressBar(maxval=len(self.experiments), \
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), progressbar.ETA()])
            bar.start()
            i = 0
            expr_num = 0
            batch_providers = []
            for e in self.experiments:
                batch_providers.extend(e.init_batch_providers(sess))
                with tf.variable_scope('experiment_' + str(expr_num)):
                    e.init_models(i%4)
                i += 1
                expr_num += 1
                bar.update(i)
            bar.finish()



            print 'Setting up optimizers'
            optimizer = SeboostOptimizer(self.experiments)

            print 'init vars'
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            print 'Reload/Dumping models'
            for e in self.experiments:
                if e.getFlagValue('epochs') > 0 and self.force_rerun == False:
                    for m in e.models:
                        m.init_from_checkpoint(sess)
                else:
                    for m in e.models:
                        m.dump_checkpoint(sess)

            sess.graph.finalize()

            # we must start queue_runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            print 'Write graph into tensorboard'
            writer = tf.summary.FileWriter('/tmp/generated_data/' + '1')
            writer.add_graph(sess.graph)
            #self.epochs = 0
            # Progress bar:
            for epoch in range(self.epochs):

                print 'Setup losses'
                models, losses, accuracies = self.remove_finished_experiments()

                # run 20 steps (full batch optimization to start with)
                print 'epoch #' + str(epoch)
                print 'Computing train and test Accuracy'
                train_error, test_error = np.zeros(len(models)), np.zeros(len(models))

                careless_batch_size = 1000
                assert (self.train_dataset_size % careless_batch_size == 0)
                assert (self.test_dataset_size % careless_batch_size == 0)
                for i in range(self.train_dataset_size / careless_batch_size):
                    train_error_feed = models[0].get_shared_feed(sess, careless_batch_size, True, models[1:])
                    train_error +=  np.array(sess.run(accuracies, feed_dict=train_error_feed))
                train_error /= float(self.train_dataset_size / careless_batch_size)
                print 'Train Accuracy = ' + str(train_error)

                for i in range(self.test_dataset_size / careless_batch_size):
                    test_error_feed = models[0].get_shared_feed(sess, careless_batch_size, False, models[1:])
                    test_error += np.array(sess.run(accuracies, feed_dict=test_error_feed))
                test_error /= float(self.test_dataset_size / careless_batch_size)
                print 'Test Accuracy = ' + str(test_error)

                print 'Dumping results....'
                self.add_experiemnts_results(train_error, test_error)
                self.dump_results()
                for m in models:
                    m.dump_checkpoint(sess)

                print 'Training'
                optimizer.run_epoch(sess=sess)

                #dump eperiments before continue to next round!

                # writer.flush()
            coord.request_stop()
            coord.join(threads)


import experiment


experiments = {}
i = 0
# for n in [1, 2, 4, 8]:
#     for h in [0, 2, 4, 8]:
#         for lr in [1.0/2**j for j in range(3,8)]:

#50000 training images and 10000

#Best conf for cifar10

def run_cifar_expr():
    ns =  [1    ]#,   1  ,    2   ,   4       ,4      ]
    hs =  [0    ]#,   1  ,    1   ,   1       ,4      ]
    lrs = [0.1  ]#,   0.1,    0.05,   0.025   ,0.025  ]

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

    runner = ExperimentRunner(experiments, force_rerun=True)
    runner.run()

run_cifar_expr()


"""
for n in [1]:
     for h in [1]:
         for lr in [0.1]:#[1.0/2**j for j in range(3,4)]:
            experiments[i] = experiment.Experiment(
            {
             'model': 'cifar10',
             'b': 128,
             'lr': lr,
             'sesop_batch_size': 1000,
             'sesop_freq': (1.0/50000.0)/300, #sesop every 300 epochs (no sesop)
             'hSize': h,
             'epochs': 250,  # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
             'nodes': n,

             #Not relevant!
             'dim': None,
             'output_dim': None,
             'dataset_size': None,
             'hidden_layers_num': None,
             'hidden_layers_size': None

             })
            i += 1

runner = ExperimentRunner(experiments, force_rerun=False)
runner.run()
"""