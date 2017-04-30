
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
import os
import threading



"""
This class manages the the background threads needed to fill
    a queue full of data.
"""
class CustomRunner(object):

    def set_data_source(self, sess, data_source_idx):
        sess.run(self.set_data_source_op[data_source_idx])

    def get_inputs(self):
        images_batch, labels_batch = self.curr_queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch

    def __init__(self, train_features, train_labels, test_features, test_labels, batch_size):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.dim = self.train_features.shape[1]

        self.label_dim = self.train_labels.shape[1]

        self.train_dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])
        self.train_dataY = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim])

        self.test_dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])
        self.test_dataY = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim])

        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.train_queue = tf.RandomShuffleQueue(shapes=[[self.dim], [self.label_dim]],
                                                 dtypes=[tf.float32, tf.float32],
                                                 capacity=6000,
                                                 min_after_dequeue=1000)

        self.test_queue = tf.FIFOQueue(shapes=[[self.dim], [self.label_dim]],
                                       dtypes=[tf.float32, tf.float32],
                                       capacity=5000)

        #0 mean test, 1 mean train
        self.data_source_idx = tf.Variable(tf.cast(1, tf.int32), trainable=False, name='data_source_idx')
        self.set_data_source_op = [tf.assign(self.data_source_idx, 0), tf.assign(self.data_source_idx, 1)]

        self.curr_queue = tf.QueueBase.from_list(tf.cast(self.data_source_idx, tf.int32),
                                       [self.test_queue, self.train_queue])

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.train_enqueue_op = self.train_queue.enqueue_many([self.train_dataX, self.train_dataY])
        self.test_enqueue_op = self.test_queue.enqueue_many([self.test_dataX, self.test_dataY])



    def train_data_iterator(self):
        """ A simple data iterator """
        batch_idx = 0
        while True:
            # shuffle labels and features
            idxs = np.arange(0, len(self.train_features))
            np.random.shuffle(idxs)
            shuf_features = self.train_features[idxs]
            shuf_labels = self.train_labels[idxs]
            for batch_idx in range(0, len(self.train_features), self.batch_size):
                images_batch = shuf_features[batch_idx:batch_idx + self.batch_size]
                images_batch = images_batch.astype("float32")
                labels_batch = shuf_labels[batch_idx:batch_idx + self.batch_size]
                yield images_batch, labels_batch

    def test_data_iterator(self):
        """ A simple data iterator """
        batch_idx = 0
        while True:
            for batch_idx in range(0, len(self.test_features), self.batch_size):
                images_batch = self.test_features[batch_idx:batch_idx + self.batch_size]
                images_batch = images_batch.astype("float32")
                labels_batch = self.test_labels[batch_idx:batch_idx + self.batch_size]
                yield images_batch, labels_batch

    def train_thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in self.train_data_iterator():
            sess.run(self.train_enqueue_op, feed_dict={self.train_dataX:dataX, self.train_dataY:dataY})

    def test_thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in self.test_data_iterator():
            sess.run(self.test_enqueue_op, feed_dict={self.test_dataX:dataX, self.test_dataY:dataY})

    def start_threads(self, sess, n_train_threads=1, n_test_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_train_threads):
            t = threading.Thread(target=self.train_thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)

        for n in range(n_test_threads):
            t = threading.Thread(target=self.test_thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class SimpleBatchProvider:
    def __init__(self, input_dim, output_dim, dataset_size, batch_size):
        self.batch_size = batch_size
        #return the same random data always.
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            DatasetManager().get_random_data(input_dim=input_dim, output_dim=output_dim, n=dataset_size)

        self.custom_runner = CustomRunner(self.training_data, self.training_labels,
                                          self.testing_data, self.testing_labels, batch_size)

    def set_data_source(self, sess, data_name='train'):
        if data_name == 'test':
            self.custom_runner.set_data_source(sess, 0)
        else:
            assert (data_name == 'train')
            self.custom_runner.set_data_source(sess, 1)

    def batch(self):
        return self.custom_runner.get_inputs()


class CifarBatchProvider:
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

