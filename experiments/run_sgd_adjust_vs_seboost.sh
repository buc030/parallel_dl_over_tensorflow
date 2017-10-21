#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID


for lr in 10 100 1 0.1
do
    #SESOP with lr update
    CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost_autoencoder.py with adaptable_learning_rate=True disable_sesop_at_all=False seboost_base_method=SGD lr=$lr &
    sleep 3

    #SGD Adjust alone
    CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost_autoencoder.py with adaptable_learning_rate=False disable_sesop_at_all=True seboost_base_method=SGD_adjust lr=$lr &

    wait

    #SESOP alone
    CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost_autoencoder.py with adaptable_learning_rate=False disable_sesop_at_all=False seboost_base_method=SGD lr=$lr &
    sleep 3

    #SGD alone
    CUDA_VISIBLE_DEVICES=3 python experiments/mnist_seboost_autoencoder.py with adaptable_learning_rate=False disable_sesop_at_all=True seboost_base_method=SGD lr=$lr &

    wait
done






