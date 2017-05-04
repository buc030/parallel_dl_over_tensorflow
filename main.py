

import matplotlib.pyplot as plt
import experiment
import experiment_results
import experiments_manager

#from experiment_runner import find_cifar_baseline, find_cifar_history
from experiment_runner import find_cifar_baseline, find_cifar_multinode, find_simple_baseline, \
    ExperimentRunner, \
    simple_multinode, simple


import argparse

parser = argparse.ArgumentParser(description='Run a simple experiment.')
parser.add_argument('-history', type=int, nargs=1, required=True, help='history number (0 for SGD)')
parser.add_argument('-n', type=int, nargs=1, required=True, help='nodes number')
parser.add_argument('-lr', type=float, nargs=1, required=True, help='Starting learning rate')
parser.add_argument('-sesop_batch_mult', type=int, nargs=1, required=True, help='Multiplier for sesop batch size')
args = parser.parse_args()




experiments = {}

experiments = simple_multinode(n=args.n[0], h=args.history[0], sesop_batch_mult=args.sesop_batch_mult[0], lr=args.lr[0])
runner = ExperimentRunner(experiments, force_rerun=True)
runner.run()

#1. Tried BFGS, doesnt change anything.
#2. Fixed a bug I inserted last week in big batch computation
#3. Changed merge algorithm
#4. Optimization in the subspace of the workers is hard, BFGS/CG cant do a good job.

print '########### DONE #################'
print '########### DONE #################'
print '########### DONE #################'
print '########### DONE #################'
print '########### DONE #################'
