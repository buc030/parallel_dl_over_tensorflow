

import experiment_results
from summary_manager import SummaryManager
from model import Model
import tensorflow as tf

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
        'lr': 0.1,
        'dataset_size': 5000,
        'model': 'simple',
        'hidden_layers_num': 1,
        'hidden_layers_size': 10,
        'nodes': 1
        }

    FLAGS_OTHER_NAMES = {'b' : 'batch_size'}

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.flags, self.results)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.flags, self.results = state

    #constructor!!
    def __init__(self, flags):
        self.flags = flags




    def __hash__(self):
        return hash(self.buildLabel())

    def __eq__(self, other):
        for k in Experiment.FLAGS_DEFAULTS.keys():
            #if k == 'epochs': continue

            if self.getFlagValue(k) != other.getFlagValue(k):
                return False
        return True

    def __str__(self):
        return 'Experiment: ' + self.buildLabel()

    def __repr__(self):
        return str(self)

    def init_models(self, batch_providers):
        assert(len(batch_providers) == self.getFlagValue('nodes'))
        #build the models and connect it to batch_providers
        self.models = []
        self.batch_providers = batch_providers
        self.results = []

        with tf.variable_scope("experiment_models") as scope:
            for batch_provider in self.batch_providers:
                #experiment, batch_provider, node_id
                self.models.append(Model(self, batch_provider, len(self.models)))
                scope.reuse_variables()

                self.results.append(experiment_results.ExperimentResults(self.buildLabel(), self.flags))

    def getFlagValue(self, name):
        if name in self.flags.keys():
            return self.flags[name]
        return Experiment.FLAGS_DEFAULTS[name]

    def buildLabel(self):
        res = ''
        for flag_name in sorted(Experiment.FLAGS_DEFAULTS.keys()):
            res += flag_name + '_' + str(self.getFlagValue(flag_name)) + '/'
        return res

    def add_train_error(self, model_idx, err):
        self.results[model_idx].trainError.append(err)

    def add_test_error(self, model_idx, err):
        self.results[model_idx].testError.append(err)
