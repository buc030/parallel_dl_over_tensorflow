import matplotlib
import numpy as np

from os import listdir
from os.path import isfile, join
import os
import re

import experiment
import matplotlib.pyplot as plt

from batch_provider import SimpleBatchProvider

import tensorflow as tf
sess = tf.InteractiveSession(target='grpc://localhost:2222')

import matplotlib.image as mpimg
import numpy as np

#input_dim, output_dim, dataset_size, batch_sizes
with tf.variable_scope('batch_providers') as scope:
    batch_prov = SimpleBatchProvider(3, 1, 5000, [2])
    scope.reuse_variables()
    batch_prov2 = SimpleBatchProvider(3, 1, 5000, [2])

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)