

import experiment_results

class Experiment:
    FLAGS_DEFAULTS = \
        {'b': 100,
        'sesop_freq': float(1) / 100,
        'hSize': 0,
        'epochs': 200,
        'dim': 10,
        'lr': 0.1
        }


    #constructor!!
    def __init__(self, flags):
        self.flags = flags
        self.results = experiment_results.ExperimentResults(self.buildLabel(), flags)

    def __hash__(self):
        return hash(self.buildLabel())

    def __eq__(self, other):
        for k in Experiment.FLAGS_DEFAULTS.keys():
            if self.getFlagValue(k) != other.getFlagValue(k):
                return False
        return True

    def __str__(self):
        return 'Experiment: ' + self.buildLabel()

    def __repr__(self):
        return str(self)

    def getFlagValue(self, name):
        if name in self.flags.keys():
            return self.flags[name]
        return Experiment.FLAGS_DEFAULTS[name]

    def buildLabel(self):
        res = ''
        for flag_name in sorted(Experiment.FLAGS_DEFAULTS.keys()):
            res += flag_name + '_' + str(self.getFlagValue(flag_name)) + '/'
        return res

    def add_train_error(self, err):
        self.results.trainError.append(err)

    def add_test_error(self, err):
        self.results.testError.append(err)
