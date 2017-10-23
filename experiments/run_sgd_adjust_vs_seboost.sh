#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID


################################################
################################################
# We start with CNN for fashion-MNIST, on two modes
# With and without dropout
################################################
################################################


#### with dropout ####
#SGD Adjust
#CUDA_VISIBLE_DEVICES=0 python experiments/mnist_seboost.py with adaptable_learning_rate=False disable_sesop_at_all=True log=/home/shai/tensorflow/logs/log1 \
#    seboost_base_method=SGD_adjust lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.0002 model=cnn disable_dropout_at_all=False disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log1 &


#SESOP Adjust
#CUDA_VISIBLE_DEVICES=1 python experiments/mnist_seboost.py with adaptable_learning_rate=True disable_sesop_at_all=False log=/home/shai/tensorflow/logs/log2 \
#    seboost_base_method=SGD lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.0002 model=cnn disable_dropout_at_all=False disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log2 &


#### without dropout ####
#SGD Adjust
#CUDA_VISIBLE_DEVICES=2 python experiments/mnist_seboost.py with adaptable_learning_rate=False disable_sesop_at_all=True  log=/home/shai/tensorflow/logs/log3 \
#    seboost_base_method=SGD_adjust lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.0002 model=cnn disable_dropout_at_all=True disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log3 &


#SESOP Adjust
#CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost.py with adaptable_learning_rate=True disable_sesop_at_all=False  log=/home/shai/tensorflow/logs/log4 \
#    seboost_base_method=SGD lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.0002 model=cnn disable_dropout_at_all=True disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log4 &



wait

################################################
################################################
# We start with CNN for fashion-MNIST, on two modes
# With and without dropout, without weight decay!
################################################
################################################

#### with dropout ####
#SGD Adjust
CUDA_VISIBLE_DEVICES=0 python experiments/mnist_seboost.py with adaptable_learning_rate=False disable_sesop_at_all=True log=/home/shai/tensorflow/logs/log9 \
    seboost_base_method=SGD_adjust lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.000 model=cnn disable_dropout_at_all=False disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log9 &


#SESOP Adjust
CUDA_VISIBLE_DEVICES=1 python experiments/mnist_seboost.py with adaptable_learning_rate=True disable_sesop_at_all=False log=/home/shai/tensorflow/logs/log10 \
    seboost_base_method=SGD lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.000 model=cnn disable_dropout_at_all=False disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log10 &


#### without dropout ####
#SGD Adjust
CUDA_VISIBLE_DEVICES=2 python experiments/mnist_seboost.py with adaptable_learning_rate=False disable_sesop_at_all=True  log=/home/shai/tensorflow/logs/log11 \
    seboost_base_method=SGD_adjust lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.000 model=cnn disable_dropout_at_all=True disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log11 &


#SESOP Adjust
CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost.py with adaptable_learning_rate=True disable_sesop_at_all=False  log=/home/shai/tensorflow/logs/log12 \
    seboost_base_method=SGD lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.000 model=cnn disable_dropout_at_all=True disable_dropout_during_sesop=True >& /home/shai/tensorflow/logs/log12 &



wait


################################################
################################################
# We continue with WRN(28, 10) for fashion-MNIST, on two modes
# With and without BN
################################################
################################################

#### with BN ####
#SGD Adjust
CUDA_VISIBLE_DEVICES=0 python experiments/mnist_seboost.py with adaptable_learning_rate=False disable_sesop_at_all=True  log=/home/shai/tensorflow/logs/log5 \
    seboost_base_method=SGD_adjust lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=True weight_decay=0.0002 model=wide-resnet disable_dropout_at_all=False disable_dropout_during_sesop=True  >& /home/shai/tensorflow/logs/log5 &


#SESOP Adjust
CUDA_VISIBLE_DEVICES=1 python experiments/mnist_seboost.py with adaptable_learning_rate=True disable_sesop_at_all=False  log=/home/shai/tensorflow/logs/log6 \
    seboost_base_method=SGD lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=True weight_decay=0.0002 model=wide-resnet disable_dropout_at_all=False disable_dropout_during_sesop=True  >& /home/shai/tensorflow/logs/log6 &


#### without BN ####
#SGD Adjust
CUDA_VISIBLE_DEVICES=2 python experiments/mnist_seboost.py with adaptable_learning_rate=False disable_sesop_at_all=True  log=/home/shai/tensorflow/logs/log7 \
    seboost_base_method=SGD_adjust lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.0002 model=wide-resnet disable_dropout_at_all=True disable_dropout_during_sesop=True  >& /home/shai/tensorflow/logs/log7 &


#SESOP Adjust
CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost.py with adaptable_learning_rate=True disable_sesop_at_all=False  log=/home/shai/tensorflow/logs/log8 \
    seboost_base_method=SGD lr=0.1 iters_per_adjust=250 update_rule=linear use_bn=False weight_decay=0.0002 model=wide-resnet disable_dropout_at_all=True disable_dropout_during_sesop=True  >& /home/shai/tensorflow/logs/log8 &





