import matplotlib
import numpy as np

from os import listdir
from os.path import isfile, join
import os
import re

import experiment
import matplotlib.pyplot as plt

from batch_provider import SimpleBatchProvider, CifarBatchProvider

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import shutil

config=tf.ConfigProto(log_device_placement=True)
sess = tf.InteractiveSession(config=config)

batch_provider = CifarBatchProvider([10])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


shutil.rmtree('/tmp/generated_data/1')
writer = tf.summary.FileWriter('/tmp/generated_data/1')
writer.add_graph(sess.graph)
writer.flush()




batch_provider.set_data_source(sess, 'train')
batch_provider.cifar_in.set_deque_batch_size(sess, 1)

img = sess.run(batch_provider.batch())
img = img[0]
img = np.reshape(img, [32, 32, 3])


imgplot = plt.imshow(img)
plt.show()