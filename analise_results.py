

import matplotlib.pyplot as plt
import experiment
import experiment_results
import experiments_manager

def display_results(experiments):

    loaded_experiments = {}
    i = 0
    for e in experiments.values():
        loaded_experiments[i] = experiments_manager.ExperimentsManager.get().load_experiment(e)
        i += 1

    comperator = experiment_results.ExperimentComperator(loaded_experiments)

    bests = {}
    j = 0
    for i in [1,2,4,8]:
        bests[j] = comperator.getBestTrainError(filter=lambda e: e.getFlagValue('nodes') == i)
        j += 1

    comperator = experiment_results.ExperimentComperator(bests)
    comperator.compare(group_by='b')
    #comperator.compare(group_by='nodes', error_type='train', filter=lambda e: e.getFlagValue('hidden_layers_num') > 0)
    plt.show()


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



display_results(experiments)
exit()

for b in [10, 100]:
    for sesop_freq in [0.01, 0.1]:
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

                    #Run in chunks of 32 (8 experiments per GPU!)
                    if i == 32:
                        print 'Running ' + str(len(experiments)) + ' experiments in parallel!'
                        display_results(experiments)
                        exit()



