

import debug_utils
import numpy as np
import experiment
import experiments_manager
from experiment_runner import ExperimentRunner, simple_multinode, simple_pbco, gans_multinode, cifar_multinode
import argparse


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

#baseline
# experiments[len(experiments)] = simple_multinode(n=1, h=0, sesop_batch_mult=12,
#                                                  lr=0.044, hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                                  DISABLE_VECTOR_BREAKING=True, NORMALIZE_DIRECTIONS=False)[0]

# # # Vanila
# experiments[len(experiments)] = simple_multinode(n=4, h=1, sesop_batch_mult=12,
#                                                  lr=0.044, hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                                  DISABLE_VECTOR_BREAKING=False, NORMALIZE_DIRECTIONS=True)[0]
# # #
# # # # With VB
# experiments[len(experiments)] = simple_multinode(n=4, h=2, sesop_batch_mult=12,
#                                                  lr=0.044, hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                                  DISABLE_VECTOR_BREAKING=False, NORMALIZE_DIRECTIONS=True)[0]
# # #
# # # # With Normaliziation
# experiments[len(experiments)] = simple_multinode(n=4, h=4, sesop_batch_mult=12,
#                                                  lr=0.044, hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                                  DISABLE_VECTOR_BREAKING=False, NORMALIZE_DIRECTIONS=True)[0]
#
# # #
# # # With both
# experiments[len(experiments)] = simple_multinode(n=4, h=8, sesop_batch_mult=12,
#                                                  lr=0.044, hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                                  DISABLE_VECTOR_BREAKING=False, NORMALIZE_DIRECTIONS=True)[0]
#
# experiments[len(experiments)] = simple_multinode(n=1, h=8, sesop_batch_mult=12,
#                                                  lr=0.044, hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                                  DISABLE_VECTOR_BREAKING=False, NORMALIZE_DIRECTIONS=True)[0]
#

#
# experiments = simple_multinode(n=args.n[0], h=args.history[0], sesop_batch_mult=args.sesop_batch_mult[0], lr=args.lr[0],
#                                hidden_layers_sizes=[6, 60, 30, 10, 1],
#                                DISABLE_VECTOR_BREAKING=args.DISABLE_VECTOR_BREAKING[0],
#                                NORMALIZE_DIRECTIONS=args.NORMALIZE_DIRECTIONS[0])


# experiments = gans_multinode(n=1, h=1)
experiments = {}
# experiments[len(experiments)] = cifar_multinode(n=1, h=0)[0]
# experiments[len(experiments)] = cifar_multinode(n=1, h=1)[0]
#experiments[len(experiments)] = cifar_multinode(n=2, h=2)[0]

experiments[len(experiments)] = cifar_multinode(n=1, h=0, fixed_bn_during_sesop=True, DISABLE_VECTOR_BREAKING=False,
                                                NORMALIZE_DIRECTIONS=True, sesop_freq=1.0/390.0, weight_decay_rate=0.0002)[0]
experiments[len(experiments)] = cifar_multinode(n=1, h=0, fixed_bn_during_sesop=True, DISABLE_VECTOR_BREAKING=False,
                                                NORMALIZE_DIRECTIONS=True, sesop_freq=1.0 / 390.0,
                                                weight_decay_rate=0.0)[0]

#
#
#
experiments[len(experiments)] = cifar_multinode(n=1, h=4, fixed_bn_during_sesop=True, DISABLE_VECTOR_BREAKING=False,
                                                NORMALIZE_DIRECTIONS=True, sesop_freq=1.0/390.0,
                                                weight_decay_rate=0.0002)[0]
experiments[len(experiments)] = cifar_multinode(n=1, h=4, fixed_bn_during_sesop=False, DISABLE_VECTOR_BREAKING=False,
                                                NORMALIZE_DIRECTIONS=True, sesop_freq=1.0/390.0,
                                                weight_decay_rate=0.0002)[0]
#

experiments[len(experiments)] = cifar_multinode(n=1, h=4, fixed_bn_during_sesop=True, DISABLE_VECTOR_BREAKING=False,
                                                NORMALIZE_DIRECTIONS=True, sesop_freq=1.0 / 390.0,
                                                weight_decay_rate=0.0)[0]
experiments[len(experiments)] = cifar_multinode(n=1, h=4, fixed_bn_during_sesop=False, DISABLE_VECTOR_BREAKING=False,
                                                NORMALIZE_DIRECTIONS=True, sesop_freq=1.0 / 390.0,
                                                weight_decay_rate=0.0)[0]


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
