

import tensorflow as tf
from seboost import SeboostOptimizer
from dataset_manager import DatasetManager
from batch_provider import BatchProvider
from model import Model
import experiments_manager
import progressbar
from experiment import Experiment

class ExperimentRunner:

    def assert_all_experiments_has_same_flag(self, flagname):
        if len(self.experiments) == 0:
            return

        val = self.experiments[0].getFlagValue(flagname)
        for e in self.experiments:
            assert (val == e.getFlagValue(flagname))

    def __init__(self, experiments, force_rerun = False):

        experiments = list(experiments.values())
        self.experiments = []

        for e in experiments:
            _e = experiments_manager.ExperimentsManager.get().load_experiment(e)
            if _e is not None:
                if force_rerun == False and len(_e.results) > 0  and len(_e.results[0].trainError) >= _e.getFlagValue('epochs'):
                    print 'Experiment ' + str(_e) + ' already ran!'
                    continue
            self.experiments.append(e)

        self.assert_all_experiments_has_same_flag('dataset_size')
        self.assert_all_experiments_has_same_flag('dim')
        self.assert_all_experiments_has_same_flag('epochs')
        self.assert_all_experiments_has_same_flag('b')
        self.assert_all_experiments_has_same_flag('sesop_freq')
        self.assert_all_experiments_has_same_flag('sesop_batch_size')

        self.bs = experiments[0].getFlagValue('b')
        self.epochs = experiments[0].getFlagValue('epochs')
        self.dataset_size = experiments[0].getFlagValue('dataset_size')
        self.dim = experiments[0].getFlagValue('dim')
        self.sesop_batch_size = experiments[0].getFlagValue('sesop_batch_size')

    def dump_results(self):
        for e in self.experiments:
            experiments_manager.ExperimentsManager.get().dump_experiment(e)

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

        # with tf.Session('grpc://' + tf_server, config=config) as sess:
        with tf.Session(config=config) as sess:


            models = []
            a,b,c,d = DatasetManager().get_random_data(dim=self.dim, n=self.dataset_size)

            max_nodes_in_experiment = max([e.getFlagValue('nodes') for e in self.experiments])


            with tf.variable_scope('batch_providers') as scope:
                batch_providers = []
                for i in range(max_nodes_in_experiment):
                    batch_providers.append(BatchProvider(a, b, c, d, [self.bs, self.sesop_batch_size, self.dataset_size]))
                    scope.reuse_variables()

            print 'building models (first e = '  + str(self.experiments[0]) + ')'
            bar = progressbar.ProgressBar(maxval=len(self.experiments), \
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), progressbar.ETA()])
            bar.start()
            i = 0
            expr_num = 0
            for e in self.experiments:

                with tf.device('/gpu:' + str(i%4)):
                    with tf.variable_scope('experiment_' + str(expr_num)):
                        e.init_models(batch_providers[0:e.getFlagValue('nodes')])
                        #All models should have the same init
                        models.extend(e.models)

                i += 1
                expr_num += 1
                bar.update(i)
            bar.finish()

            print 'Setup losses'
            losses = [m.loss() for m in models]

            print 'Setting up optimizers'
            optimizer = SeboostOptimizer(self.experiments)

            print 'init vars'
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())


            sess.graph.finalize()

            # we must start queue_runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)



            #All models have the same batch provider!
            set_full_batch_training = [models[0].get_batch_provider().set_batch_size_op(self.dataset_size)] + \
                [models[0].get_batch_provider().set_training_op(True)]

            set_full_batch_testing = [models[0].get_batch_provider().set_batch_size_op(self.dataset_size)] + \
                [models[0].get_batch_provider().set_training_op(False)]

            set_training = [models[0].get_batch_provider().set_batch_size_op(self.bs)] + \
                [models[0].get_batch_provider().set_training_op(True)]

            print 'Write graph into tensorboard'
            writer = tf.summary.FileWriter('/tmp/generated_data/' + '1')
            writer.add_graph(sess.graph)
            #self.epochs = 0
            # Progress bar:
            for epoch in range(self.epochs):

                # run 20 steps (full batch optimization to start with)
                print 'epoch #' + str(epoch)
                print 'Computing train error'
                sess.run(set_full_batch_training)
                total_losses = sess.run(losses)
                print 'Train error = ' + str(total_losses)
                i = 0
                for e in self.experiments:
                    for model_idx in range(len(e.models)):
                        e.add_train_error(model_idx, total_losses[i])
                        i += 1


                print 'Computing test error'
                sess.run(set_full_batch_testing)
                total_losses = sess.run(losses)
                i = 0
                print 'Test error = ' + str(total_losses)
                for e in self.experiments:
                    for model_idx in range(len(e.models)):
                        e.add_test_error(model_idx, total_losses[i])
                        i += 1

                print 'Dumping results....'
                self.dump_results()

                print 'Set Training (setting batch size and train set)'
                sess.run(set_training)
                print 'Training'
                optimizer.run_epoch(sess=sess)

                #dump eperiments before continue to next round!

                # writer.flush()
            coord.request_stop()
            coord.join(threads)


import experiment

experiments = {}
"""
i = 0

for b in [10, 100]:
    # make sure we break here...
    if len(experiments) > 0:
        print 'Running ' + str(len(experiments)) + ' experiments in parallel!'
        runner = ExperimentRunner(experiments, force_rerun=False)
        runner.run()

    experiments = {}
    i = 0

    for sesop_freq in [0.01, 0.1]:
        #make sure we break here...
        if len(experiments) > 0:
            print 'Running ' + str(len(experiments)) + ' experiments in parallel!'
            runner = ExperimentRunner(experiments, force_rerun=False)
            runner.run()

        experiments = {}
        i = 0

        for hidden_layers_num in [1, 2, 4, 8]:
            for h in [2**z for z in range(0, 8)]:
                for lr in [float(1)/2**j for j in range(4, 8)]:
                #for lr in [1.0/32, 1.0/64, 1.0/128]:
                    experiments[i] = experiment.Experiment(
                        {'b': b,
                         'sesop_freq': sesop_freq,
                         'hSize': h,
                         'epochs': 100, #saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
                         'dim': 10,
                         'lr': lr,
                         'dataset_size': 5000,
                         'model': 'simple',
                         'hidden_layers_num': hidden_layers_num,
                         'hidden_layers_size': 10
                         })
                    i += 1

                    #Run in chunks of 16 (4 experiments per GPU!)
                    if i == 4:
                        print 'Running ' + str(len(experiments)) + ' experiments in parallel!'
                        runner = ExperimentRunner(experiments, force_rerun=False)
                        runner.run()
                        experiments = {}
                        i = 0
"""

experiments = {}
i = 0
for n in [1, 2, 4, 8]:
    for h in [0, 2, 4]:
        for lr in [1.0/2**j for j in range(3,7)]:
# for n in [1]:
#      for h in [5]:
#          for lr in [1.0/2**j for j in range(3,4)]:
            experiments[i] = experiment.Experiment(
            {'b': 10,
             'sesop_batch_size': 300,
             'sesop_freq': 10.0/5000.0, #sesop once an epoch
             'hSize': h,
             'epochs': 50,  # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
             'dim': 10,
             'lr': lr,
             'dataset_size': 5000,
             'model': 'simple',
             'hidden_layers_num': 3,
             'hidden_layers_size': 10,
             'nodes': n
             })
            i += 1

runner = ExperimentRunner(experiments, force_rerun=False)
runner.run()

import experiment_results

comperator = experiment_results.ExperimentComperator(experiments)


import matplotlib.pyplot as plt

comperator.compare(group_by='b', error_type='train')
plt.show()
