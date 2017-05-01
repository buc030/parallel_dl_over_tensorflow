

import matplotlib.pyplot as plt
import experiment
import experiment_results
import experiments_manager

#from experiment_runner import find_cifar_baseline, find_cifar_history
from experiment_runner import find_cifar_baseline, find_cifar_multinode, find_simple_baseline, \
    ExperimentRunner, \
    simple_multinode, simple





experiments = {}

############ SIMPLE (first make sure we can overfit the data) #################
# experiments = simple()
# runner = ExperimentRunner(experiments, force_rerun=True)
# runner.run()


############ BASELINE #################
# experiments = find_simple_baseline()
# runner = ExperimentRunner(experiments, force_rerun=True)
# runner.run()



############# HISTORY  #################
# experiments = simple_with_history_baseline(h=8, sesop_batch_mult=8)
# runner = ExperimentRunner(experiments, force_rerun=True)
# runner.run()


############ MULTI NODE #################

experiments = simple_multinode(n=2, h=4, sesop_batch_mult=5)
runner = ExperimentRunner(experiments, force_rerun=True)
runner.run()

#1. Tried BFGS, doesnt change anything.
#2. Fixed a bug I inserted last week in big batch computation
#3. Changed merge algorithm


print '########### DONE #################'
print '########### DONE #################'
print '########### DONE #################'
print '########### DONE #################'
print '########### DONE #################'
