

import experiment_results
from summary_manager import SummaryManager
from model import SimpleModel, MnistModel, CifarModel
import tensorflow as tf
from batch_provider import SimpleBatchProvider, CifarBatchProvider
from experiments_manager import ExperimentsManager
import debug_utils
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
        'lr': 0.1,
        'dataset_size': 5000,
        'model': 'simple',
        'hidden_layers_sizes': None,
        'nodes': 1,
        'base_optimizer': 'sgd',
        'num_residual_units': 9,
        'fixed_dropout_during_sesop': False,
        'fixed_bn_during_sesop': False,
        'sesop_batch_mult' : 1,
        'subspace_optimizer' : 'BFGS',
        'DISABLE_VECTOR_BREAKING' : True,
        'NORMALIZE_DIRECTIONS' : False,
        'learning_rate_per_node': False
    }

    FLAGS_ALIASES = {'batch_size' : 'b'}

    @classmethod
    def flag_names_iterator(cls):
        for k in Experiment.FLAGS_DEFAULTS.keys():
            yield k

    def has_data(self):
        return len(self.results) > 0 and len(self.results[0].trainError) > 0

    def getCanonicalName(self, name):
        if name not in Experiment.FLAGS_ALIASES:
            return name
        return Experiment.FLAGS_ALIASES[name]


    def getFlagValue(self, name):
        name = self.getCanonicalName(name)
        if name in self.flags.keys():
            return self.flags[name]
        return Experiment.FLAGS_DEFAULTS[name]

    #if I added a flag after the run took place, but i put the flag value of the run as defualt,
    #then when loading the run, we will load it without that flag, but when we will look up this run in the metadata
    #it will match since it uses 'getFlagValue' that checks in FLAGS_DEFAULTS
    def __eq__(self, other):
        for k in Experiment.FLAGS_DEFAULTS.keys():
            if k == 'epochs': continue

            if self.getFlagValue(k) != other.getFlagValue(k):
                return False
        return True



    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.flags, self.results)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        flags, results = state
        self.__init__(flags)
        self.results = results

    #constructor!!
    def __init__(self, flags):
        self.flags = flags

        self.bs, self.sesop_batch_size = flags['b'], flags['sesop_batch_size']
        self.sesop_batch_mult = self.getFlagValue('sesop_batch_mult')


        self.results = []

        if self.getFlagValue('model') == 'simple':
            self.train_dataset_size = self.getFlagValue('dataset_size')
            self.test_dataset_size = self.train_dataset_size
        elif self.getFlagValue('model') == 'cifar10':
            self.train_dataset_size = 40000
            self.sesop_dataset_size = 10000
            self.test_dataset_size = 10000
        else:
            assert (False)

    def __hash__(self):
        return hash(self.buildLabel())

    def __str__(self):
        return 'Experiment: ' + self.buildLabel()

    def __repr__(self):
        return str(self)

    def getInputDim(self):
        hidden_layers_sizes = self.getFlagValue('hidden_layers_sizes')
        input_dim = hidden_layers_sizes[0]
        return input_dim

    def getOutputDim(self):
        hidden_layers_sizes = self.getFlagValue('hidden_layers_sizes')
        output_dim = hidden_layers_sizes[-1]
        return output_dim

    def get_model_tensorboard_dir(self, model_idx):
        return ExperimentsManager.get().get_experiment_model_tensorboard_dir(self, model_idx)


    def get_model_checkpoint_dir(self, model_idx):
        return ExperimentsManager.get().get_experiment_model_checkpoint_dir(self, model_idx)


    def init_models(self, gpu, batch_providers):
        assert(len(batch_providers) >= self.getFlagValue('nodes'))
        self.models = []
        #build the models and connect it to batch_providers
        with tf.variable_scope("experiment_models") as scope:
            for i in range(self.getFlagValue('nodes')):
                with tf.device('/gpu:' + str(gpu % 4)):
                    #experiment, batch_provider, node_id
                    model = None
                    if self.getFlagValue('model') == 'simple':
                        model = SimpleModel(self, batch_providers[i], len(self.models))
                    elif self.getFlagValue('model') == 'mnist':
                        model = MnistModel(self, batch_providers[i], len(self.models))
                    elif self.getFlagValue('model') == 'cifar10':
                        model = CifarModel(self, batch_providers[i], len(self.models))

                    assert(model is not None)

                    model.gpu = gpu % 4
                    self.models.append(model)
                    #scope.reuse_variables()

                    self.results.append(experiment_results.ExperimentResults(self.buildLabel(), self.flags))
                    gpu += 1

        return self.getFlagValue('nodes')


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
        if len(self.results) == 0 or len(self.results[0].trainError) == 0:
            return 0

        return len(self.results[0].trainError) - 1

    def start_new_subspace_optimization(self):
        self.results[0].loss_during_supspace_optimization.append([])
        self.results[0].grad_norm_during_supspace_optimization.append([])


    def push_sgd_epoch(self, model_idx):
        self.results[model_idx].sgd_epoch_grad_norms.append([])
        self.results[model_idx].sgd_epoch_weights_norms.append([])
        self.results[model_idx].sgd_epoch_input_norms.append([])

    def add_sgd_iter_input_norm(self, model_idx, norm):
        self.results[model_idx].sgd_epoch_input_norms[-1].append(norm)

    def add_sgd_iter_weight_norm(self, model_idx, norm):
        self.results[model_idx].sgd_epoch_weights_norms[-1].append(norm)

    def add_sgd_iter_grad_norm(self, model_idx, norm):
        self.results[model_idx].sgd_epoch_grad_norms[-1].append(norm)

    def add_loss_during_supspace_optimization(self, loss):
        self.results[0].loss_during_supspace_optimization[-1].append(loss)

    def add_grad_norm_during_supspace_optimization(self, loss):
        self.results[0].grad_norm_during_supspace_optimization[-1].append(loss)


    def add_iteration_train_error(self, model_idx, err):
        self.results[model_idx].trainErrorPerItereation.append(err)

    def add_debug_sesop(self, model_idx, before, after):
        self.results[model_idx].debug_sesop_before.append(before)
        self.results[model_idx].debug_sesop_after.append(after)

    def add_debug_sesop_on_sesop_batch(self, model_idx, before, after):
        self.results[model_idx].debug_sesop_on_sesop_batch_before.append(before)
        self.results[model_idx].debug_sesop_on_sesop_batch_after.append(after)



    def add_train_error(self, model_idx, err):
        self.results[model_idx].trainError.append(err)

    def add_test_error(self, model_idx, err):
        self.results[model_idx].testError.append(err)
