

import matplotlib.pyplot as plt
import experiment
import experiment_results
import experiments_manager

#from experiment_runner import find_cifar_baseline, find_cifar_history
from experiment_runner import find_cifar_baseline, find_cifar_multinode, \
    find_simple_baseline, simple_multinode, simple
def display_results(experiments):

    loaded_experiments = {}
    i = 0
    for e in experiments.values():
        loaded_experiments[i] = experiments_manager.ExperimentsManager.get().load_experiment(e)
        print loaded_experiments[i]
        print 'loaded_experiments[i] = ' + str(loaded_experiments[i].results)
        i += 1

    print str(loaded_experiments[0].results)
    comperator = experiment_results.ExperimentComperator(loaded_experiments)
    comperator.set_y_logscale(True)


    # comperator.compare(group_by='lr', error_type='train_and_test')
    # plt.show()

    # comperator.compare(group_by='dataset_size', error_type='trainPerIteration')
    # plt.show()

    # comperator.compare(group_by='nodes', error_type='train')
    # plt.show()
    comperator.compare(group_by='hSize', error_type='test')
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


experiments = {}

#experiments = find_simple_baseline()
# for h in [1, 2, 4, 8, 16]:
#     experiments[len(experiments)] = simple_with_history_baseline(h=h, sesop_batch_mult=5)[0]

########## 1 node ###########
experiments[len(experiments)] = simple_multinode(n=1, h=0, sesop_batch_mult=5)[0]
experiments[len(experiments)] = simple_multinode(n=1, h=1, sesop_batch_mult=5)[0]
experiments[len(experiments)] = simple_multinode(n=1, h=2, sesop_batch_mult=5)[0]
experiments[len(experiments)] = simple_multinode(n=1, h=4, sesop_batch_mult=5)[0]

########## 2 node ###########
experiments[len(experiments)] = simple_multinode(n=2, h=0, sesop_batch_mult=5)[0]
experiments[len(experiments)] = simple_multinode(n=2, h=1, sesop_batch_mult=5)[0]
experiments[len(experiments)] = simple_multinode(n=2, h=2, sesop_batch_mult=5)[0]
experiments[len(experiments)] = simple_multinode(n=2, h=4, sesop_batch_mult=5)[0]


########## 4 node ###########
experiments[len(experiments)] = simple_multinode(n=4, h=4, sesop_batch_mult=5)[0]


# experiments[len(experiments)] = simple_multinode(n=4, h=2, sesop_batch_mult=5)[0]
# experiments[len(experiments)] = simple_multinode(n=2, h=0, sesop_batch_mult=5)[0]
# experiments[len(experiments)] = simple_multinode(n=2, h=1, sesop_batch_mult=5)[0]
# experiments[len(experiments)] = simple_multinode(n=2, h=2, sesop_batch_mult=5)[0]
# experiments[len(experiments)] = simple_multinode(n=2, h=4, sesop_batch_mult=5)[0]



display_results(experiments)