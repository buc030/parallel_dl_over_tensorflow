

import debug_utils
import argparse
import numpy as np
import experiment
import experiments_manager
from experiment_runner import ExperimentRunner, simple_multinode


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Run a simple experiment.')
parser.add_argument('-history', type=int, nargs=1, required=True, help='history number (0 for SGD)')
parser.add_argument('-n', type=int, nargs=1, required=True, help='nodes number')
parser.add_argument('-lr', type=float, nargs=1, required=True, help='Starting learning rate')
parser.add_argument('-sesop_batch_mult', type=int, nargs=1, required=True, help='Multiplier for sesop batch size')

parser.add_argument('-NORMALIZE_DIRECTIONS', type=str2bool, nargs=1, required=True, help='')
parser.add_argument('-DISABLE_VECTOR_BREAKING', type=str2bool, nargs=1, required=True, help='')

# args = parser.parse_args()


experiments = {}






for h in [1, 2, 4, 8, 16, 32, 64]:
    experiments[len(experiments)] = simple_multinode(n=1, h=h, sesop_batch_mult=10, lr=0.7425, hidden_layers_sizes=[6, 60, 30, 10, 1])[0]


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
