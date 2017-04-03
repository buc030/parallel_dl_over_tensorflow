
import numpy
import tensorflow as tf

from dataset_manager import DatasetManager
import cifar_input
import Queue
from threading import Thread

"""
Every subclass need to implment
        self.train_pipes = {}
        self.test_pipes = {}

"""
class BatchProvider(object):

    def get_queue_index(self, batch_size, is_training):
        return self.b_train_to_index[(batch_size, is_training)]

    def get_queue_by_index(self, index):
        batch_size, is_training = self.index_to_b_train[index]
        if is_training:
            return self.train_pipes[batch_size]

        return self.test_pipes[batch_size]

    def get_queues(self):
        return [self.get_queue_by_index(i) for i in range(len(self.batch_sizes)*2)]

    def set_source(self, sess, batch_size, is_training):
        # Have the que runner do nothing by setting batch_size = -1
        sess.run(tf.assign(self.que_chooser, self.get_queue_index(batch_size, is_training)))


    def __init__(self, batch_sizes):

        self.batch_sizes = batch_sizes
        #different solution:
        self.b_train_to_index = {}
        self.index_to_b_train = {}
        i = 0
        for b in batch_sizes:
            self.b_train_to_index[(b, True)] = i
            self.index_to_b_train[i] = (b, True)
            i += 1
            self.b_train_to_index[(b, False)] = i
            self.index_to_b_train[i] = (b, False)
            i += 1

        self.que_chooser = tf.Variable(0, trainable=False, name='que_chooser')
        print 'self.get_queues() = ' + str(self.get_queues()[0])
        self._batch = tf.QueueBase.from_list(self.que_chooser, self.get_queues()).dequeue()

    def batch(self):
        return self._batch

    def next_batch(self, sess, batch_size, is_train):

        if is_train:
            return sess.run(self.train_pipes[batch_size])

        return sess.run(self.test_pipes[batch_size])


class CifarBatchProvider(BatchProvider):
    def __init__(self, batch_sizes):

        self.train_pipes = {}
        self.test_pipes = {}

        for b in batch_sizes:
            self.train_pipes[b] = cifar_input.build_input('cifar10', 'CIFAR_data/cifar-10-batches-bin/data_batch*', b, 'train')
            self.test_pipes[b] = cifar_input.build_input('cifar10', 'CIFAR_data/cifar-10-batches-bin/test_batch.bin', b, 'test')

        super(CifarBatchProvider, self).__init__(batch_sizes)



class SimpleBatchProvider(BatchProvider):

    ##private
    def create_batching_pipeline(self, full_input, full_label, b):
        input, label = tf.train.slice_input_producer([full_input, full_label], name='slicer', shuffle=True, seed=1)
        batched_input, batched_labels = tf.train.batch([input, label], batch_size=b, name='batcher', capacity=2*max(self.batch_sizes))


        queue = tf.FIFOQueue(capacity=1, dtypes=tf.float32, name='staging_que')
        enque_op = queue.enqueue([batched_input, batched_labels])
        tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enque_op] * 1))

        return queue


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



