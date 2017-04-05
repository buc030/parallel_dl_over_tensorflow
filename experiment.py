

import experiment_results
from summary_manager import SummaryManager
from model import SimpleModel, MnistModel, CifarModel
import tensorflow as tf
from batch_provider import SimpleBatchProvider, CifarBatchProvider
from experiments_manager import ExperimentsManager

#This defines the metaparameters and
#store the results
class Experiment:
    FLAGS_DEFAULTS = \
        {
        'b': 100,
        'sesop_batch_size': 200,
        'sesop_freq': float(1) / 100,
        'hSize': 0,
        'epochs': 200,
        'dim': 10,
        'output_dim': 1,
        'lr': 0.1,
        'dataset_size': 5000,
        'model': 'simple',
        'hidden_layers_num': 1,
        'hidden_layers_size': 10,
        'nodes': 1,
        'optimizer': 'sgd'
        }

    FLAGS_OTHER_NAMES = {'b' : 'batch_size'}

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.flags, self.results)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.flags, self.results = state
        self.__init__(self.flags)

    #constructor!!
    def __init__(self, flags):
        self.flags = flags

        self.bs, self.sesop_batch_size = flags['b'], flags['sesop_batch_size']

        if self.getFlagValue('model') == 'simple':
            self.train_dataset_size = 5000
            self.test_dataset_size = 5000
        elif self.getFlagValue('model') == 'cifar10':
            self.train_dataset_size = 50000
            self.test_dataset_size = 10000
        else:
            assert (False)

    def __hash__(self):
        return hash(self.buildLabel())

    def __eq__(self, other):
        for k in Experiment.FLAGS_DEFAULTS.keys():
            if k == 'epochs': continue

            if self.getFlagValue(k) != other.getFlagValue(k):
                return False
        return True

    def __str__(self):
        return 'Experiment: ' + self.buildLabel()

    def __repr__(self):
        return str(self)


    def get_model_tensorboard_dir(self, model_idx):
        return ExperimentsManager.get().get_experiment_model_tensorboard_dir(self, model_idx)


    def get_model_checkpoint_dir(self, model_idx):
        return ExperimentsManager.get().get_experiment_model_checkpoint_dir(self, model_idx)


    def init_models(self, gpu, batch_providers):
        assert(len(batch_providers) >= self.getFlagValue('nodes'))
        #build the models and connect it to batch_providers
        self.models = []
        self.results = []
        with tf.device('/gpu:' + str(gpu % 4)):
            with tf.variable_scope("experiment_models") as scope:
                for i in range(self.getFlagValue('nodes')):
                    #experiment, batch_provider, node_id
                    model = None
                    if self.getFlagValue('model') == 'simple':
                        model = SimpleModel(self, batch_providers[i], len(self.models))
                    elif self.getFlagValue('model') == 'mnist':
                        model = MnistModel(self, batch_providers[i], len(self.models))
                    elif self.getFlagValue('model') == 'cifar10':
                        model = CifarModel(self, batch_providers[i], len(self.models))

                    assert(model is not None)

                    self.models.append(model)
                    scope.reuse_variables()

                    self.results.append(experiment_results.ExperimentResults(self.buildLabel(), self.flags))

    def getFlagValue(self, name):
        if name in self.flags.keys():
            return self.flags[name]
        return Experiment.FLAGS_DEFAULTS[name]

    def getDatasetSize(self):
        if self.getFlagValue('model') == 'simple':
            self.train_dataset_size = self.getFlagValue('dataset_size')
            self.test_dataset_size = self.getFlagValue('dataset_size')
        elif self.getFlagValue('model') == 'cifar10':
            self.train_dataset_size = 50000
            self.test_dataset_size = 10000
        else:
            assert (False)

        return self.train_dataset_size, self.test_dataset_size

    def buildLabel(self):
        res = ''
        for flag_name in sorted(Experiment.FLAGS_DEFAULTS.keys()):
            res += flag_name + '_' + str(self.getFlagValue(flag_name)) + '/'
        return res

    def get_number_of_ran_iterations(self):
        #TODO: assert that all models has same length
        if len(self.results) == 0:
            return 0

        return len(self.results[0].trainErrorPerItereation)

    def get_number_of_ran_epochs(self):
        # TODO: assert that all models has same length
        if len(self.results) == 0:
            return 0

        return len(self.results[0].trainError)

    def add_iteration_train_error(self, model_idx, err):
        self.results[model_idx].trainErrorPerItereation.append(err)

    def add_train_error(self, model_idx, err):
        self.results[model_idx].trainError.append(err)

    def add_test_error(self, model_idx, err):
        self.results[model_idx].testError.append(err)
