
import numpy
import tensorflow as tf

from dataset_manager import DatasetManager
import cifar_input
import Queue
from threading import Thread
import utils
from tensorflow.python.ops import data_flow_ops
from tf_utils import StackedBatches
import numpy as np

"""
Every subclass need to implment
        self.train_pipes = {}
        self.test_pipes = {}

"""
class BatchProvider(object):
    def __init__(self, batch_sizes):

        self.batch_sizes = batch_sizes


        self.set_source_ops = {}
        for b in batch_sizes:
            # 0 means test
            # 1 means train
            # 2 means sesop
            for val in [0, 1, 2]:
                self.set_source_ops[(b, val)] = [tf.assign(self.batch_size_chooser, b), tf.assign(self.is_train_chooser, val)]

        self._batch = self.pipe

    def batch(self):
        return self._batch

    def set_source(self, sess, batch_size, is_training):
        # Have the que runner do nothing by setting batch_size = -1
        sess.run(self.set_source_ops[(batch_size, is_training)])



class CifarBatchProvider(BatchProvider):
    def __init__(self, batch_sizes, train_threads):

        #dataset, data_path, test_path, batch_size, max_batch_size, is_training, num_threads
        self.batch_size_chooser = tf.Variable(batch_sizes[0], trainable=False, name='batch_size_chooser')

        #0 means test
        #1 means train
        #2 means sesop
        self.is_train_chooser = tf.Variable(tf.cast(1, tf.int32), trainable=False, name='is_train_chooser')

        self.cifar_in = cifar_input.CifarInput()
        self.pipe = self.cifar_in.build_input('cifar10', 'CIFAR_data/cifar-10-batches-bin/data_batch*',\
                'CIFAR_data/cifar-10-batches-bin/test_batch.bin',\
                self.batch_size_chooser, max_batch_size=max(batch_sizes), is_training=self.is_train_chooser, num_threads=train_threads)


        super(CifarBatchProvider, self).__init__(batch_sizes)

import os

class SimpleBatchProvider(BatchProvider):

    ##private
    def create_train_pipeline(self, full_input, full_label, b):
        pid = os.getpid()
        #tensorflow is sharing these data structures between different runs!
        input, label = tf.train.slice_input_producer([full_input, full_label], name='slicer_train', shuffle=True, seed=143)
        batched_input, batched_labels = tf.train.batch([input, label], batch_size=b, name='batcher_train', capacity=16*max(self.batch_sizes), num_threads=4)

        return batched_input, batched_labels

    def create_test_pipeline(self, full_input, full_label, b):
        pid = os.getpid()

        input, label = tf.train.slice_input_producer([full_input, full_label], name='slicer_test', shuffle=False, seed=143)
        batched_input, batched_labels = tf.train.batch([input, label], batch_size=b, name='batcher_test',
                                                       capacity=max(self.batch_sizes))

        return batched_input, batched_labels

    ##public
    #User has to say infornt what batch sizes does he wants to support!
    def __init__(self, input_dim, output_dim, dataset_size, batch_sizes):
        self.batch_sizes = batch_sizes
        #return the same random data always.
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            DatasetManager().get_random_data(input_dim=input_dim, output_dim=output_dim, n=dataset_size)

        # dataset, data_path, test_path, batch_size, max_batch_size, is_training, num_threads
        self.batch_size_chooser = tf.Variable(batch_sizes[0], trainable=False, name='batch_size_chooser')

        # 0 means test
        # 1 means train
        # 2 means sesop
        self.is_train_chooser = tf.Variable(tf.cast(1, tf.int32), trainable=False, name='is_train_chooser')


        with tf.name_scope('simple_data_provider'):
            self.sub_init()

        super(SimpleBatchProvider, self).__init__(batch_sizes)

    def sub_init(self):
        self.full_train_input = tf.Variable(name='train_dataset_x',
                                                initial_value=self.training_data,
                                                trainable=False)
        self.full_train_labels = tf.Variable(name='train_dataset_y',
                                                 initial_value=self.training_labels,
                                                 trainable=False)

        self.full_test_input = tf.Variable(name='test_dataset_x',
                                               initial_value=self.testing_data,
                                               trainable=False)

        #print self.full_test_input
        #print 'self.testing_data = ' + str(self.testing_data.shape)
        self.full_test_labels = tf.Variable(name='test_dataset_y',
                                                initial_value=self.testing_labels,
                                                trainable=False)

        self.test_pipe = self.create_test_pipeline(self.full_test_input, self.full_test_labels, self.batch_size_chooser)
        self.train_pipe = self.create_train_pipeline(self.full_train_input, self.full_train_labels, self.batch_size_chooser)
        #self.pipe = self.train_pipe
        self.pipe = tf.cond(pred=tf.equal(self.is_train_chooser, 0), fn1=lambda : self.test_pipe, fn2=lambda : self.train_pipe)





