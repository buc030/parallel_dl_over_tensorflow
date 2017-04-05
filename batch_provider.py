
import numpy
import tensorflow as tf

from dataset_manager import DatasetManager
import cifar_input
import Queue
from threading import Thread
import utils
from tensorflow.python.ops import data_flow_ops
from tf_utils import StackedBatches
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
            for val in [False, True]:
                self.set_source_ops[(b, val)] = [tf.assign(self.batch_size_chooser, b), tf.assign(self.is_train_chooser, val)]

        self._batch = self.train_pipe

    def batch(self):
        return self._batch

    def set_source(self, sess, batch_size, is_training):
        # Have the que runner do nothing by setting batch_size = -1
        sess.run(self.set_source_ops[(batch_size, is_training)])



class CifarBatchProvider(BatchProvider):
    def __init__(self, batch_sizes, train_threads):



        #dataset, data_path, test_path, batch_size, max_batch_size, is_training, num_threads
        self.batch_size_chooser = tf.Variable(batch_sizes[0], trainable=False, name='batch_size_chooser')
        self.is_train_chooser = tf.Variable(False, trainable=False, name='is_train_chooser')

        self.train_pipe = cifar_input.build_input('cifar10', 'CIFAR_data/cifar-10-batches-bin/data_batch*',\
                'CIFAR_data/cifar-10-batches-bin/test_batch.bin',\
                self.batch_size_chooser, max_batch_size=max(batch_sizes), is_training=self.is_train_chooser, num_threads=train_threads)

        #
        # self.test_pipe = cifar_input.build_input('cifar10', 'CIFAR_data/cifar-10-batches-bin/test_batch.bin', \
        #                                           self.batch_size_chooser, max_batch_size=max(batch_sizes), mode='train', num_threads=1)
        super(CifarBatchProvider, self).__init__(batch_sizes)



class SimpleBatchProvider(BatchProvider):

    ##private
    def create_batching_pipeline(self, full_input, full_label, b):
        input, label = tf.train.slice_input_producer([full_input, full_label], name='slicer', shuffle=True, seed=1)
        batched_input, batched_labels = tf.train.batch([input, label], batch_size=b, name='batcher', capacity=2*max(self.batch_sizes))

        return batched_input, batched_labels


    ##public
    #User has to say infornt what batch sizes does he wants to support!
    def __init__(self, input_dim, output_dim, dataset_size, batch_sizes):

        #return the same random data always.
        training_data, testing_data, training_labels, testing_labels = \
            DatasetManager().get_random_data(input_dim=input_dim, output_dim=output_dim, n=dataset_size)

        with tf.name_scope('simple_data_provider'):
            self.sub_init(training_data, testing_data, training_labels, testing_labels, batch_sizes)

        super(SimpleBatchProvider, self).__init__(batch_sizes)

    def sub_init(self, training_data, testing_data, training_labels, testing_labels, batch_sizes):
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            training_data, testing_data, training_labels, testing_labels

        self.batch_sizes = batch_sizes


        self.full_train_input = tf.get_variable(name='train_dataset_x',
                                                initializer=lambda shape, dtype, partition_info: training_data,
                                                shape=training_data.shape, trainable=False)
        self.full_train_labels = tf.get_variable(name='train_dataset_y',
                                                 initializer=lambda shape, dtype, partition_info: training_labels,
                                                 shape=training_labels.shape, trainable=False)

        self.full_test_input = tf.get_variable(name='test_dataset_x',
                                               initializer=lambda shape, dtype, partition_info: testing_data,
                                               shape=testing_data.shape, trainable=False)
        self.full_test_labels = tf.get_variable(name='test_dataset_y',
                                                initializer=lambda shape, dtype, partition_info: testing_labels,
                                                shape=testing_labels.shape, trainable=False)

        self.train_pipes, self.test_pipes = {}, {}
        for b in batch_sizes:
            self.train_pipes[b] = self.create_batching_pipeline(self.full_train_input, self.full_train_labels, b)
            self.test_pipes[b] = self.create_batching_pipeline(self.full_test_input, self.full_test_labels, b)



