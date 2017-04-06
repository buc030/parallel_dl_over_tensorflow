

import matplotlib.pyplot as plt
import experiment
import experiment_results
import experiments_manager

#from experiment_runner import find_cifar_baseline, find_cifar_history
import experiment_runner
def display_results(experiments):

    loaded_experiments = {}
    i = 0
    for e in experiments.values():
        loaded_experiments[i] = experiments_manager.ExperimentsManager.get().load_experiment(e)
        i += 1

    print str(loaded_experiments[0].results)
    comperator = experiment_results.ExperimentComperator(loaded_experiments)

    comperator.compare(group_by='b', error_type='test')
    plt.show()
    comperator.compare(group_by='b', error_type='train')
    plt.show()


    bests = {}
    j = 0
    for i in [1,2,4,8]:
        bests[j] = comperator.getBestTrainError(filter=lambda e: e.getFlagValue('nodes') == i)# and e.getFlagValue('lr') == 1.0/2**7)
        j += 1

    comperator = experiment_results.ExperimentComperator(bests)
    comperator.compare(group_by='b')
    #comperator.compare(group_by='nodes', error_type='train', filter=lambda e: e.getFlagValue('hidden_layers_num') > 0)
    plt.show()


experiments = experiment_runner.find_cifar_baseline()
display_results(experiments)