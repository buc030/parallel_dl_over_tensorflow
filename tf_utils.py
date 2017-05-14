
import tensorflow as tf
import progressbar
from progressbar import ProgressBar, Percentage, Bar, ETA
from time import sleep

import threading
import numpy as np

#feed_dicts hold mapping for the input data, each entry hold a batch.
#additional_feed_dict hold mapping to variables or something like that.
#
def avarge_on_feed_dicts(sess, target_ops, feed_dicts, additional_feed_dict={}):
    feed_dict = additional_feed_dict
    feed_dict.update(feed_dicts[0])
    res = sess.run(target_ops, feed_dict=feed_dict)

    for i in range(1, len(feed_dicts)):
        feed_dict.update(feed_dicts[i])
        temp = sess.run(target_ops, feed_dict=feed_dict)
        for j in range(len(target_ops)):
            res[j] += temp[j]

    for j in range(len(target_ops)):
        res[j] /= len(feed_dicts)

    return res



#returns an op that concat op_func() n times
#op_func is a function that creates an op
def chain_training_step(op_func, n):
    print 'Creating ' + str(n) + ' chained training steps'
    if n == 0:
        return tf.no_op()
    bar = progressbar.ProgressBar(maxval=n, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    res = tf.no_op()
    for i in range(n):
        with tf.control_dependencies([res]):
            step = op_func()
            res = tf.group(res, step)
        bar.update(i + 1)
    bar.finish()
    return res






class StackedBatches:
    def __init__(self):
        self.batches = []
        self.batch_sizes = []

    def add_batch(self, batch, batch_size):
        self.batches.append(batch)
        self.batch_sizes.append(batch_size)

    def build_stacked_batches(self):
        self._stack_batches = tf.concat(values=self.batches, axis=0)

        self.batch_sizes_2_indexes = {}
        for target_batch_size in self.batch_sizes:
            first_idx = 0
            for b in self.batch_sizes:
                if target_batch_size == b:
                    self.batch_sizes_2_indexes[target_batch_size] = first_idx
                    break
                first_idx += b
            assert(target_batch_size  in self.batch_sizes_2_indexes)

    def get_batch_by_batchsize_op(self, batch_size):
        index = self.batch_sizes_2_indexes[batch_size]
        return self._stack_batches[index : index+batch_size]



"""
This class manages the the background threads needed to fill
    a queue full of data.
"""
class CustomRunner(object):

    def set_data_source(self, sess, data_source_idx):
        sess.run(self.set_data_source_op[data_source_idx])

    def set_deque_batch_size(self, sess, new_batch_size):
        sess.run(self.set_deque_batch_size_op, feed_dict={self.batch_size_tf_placeholder : new_batch_size})

    def get_inputs(self):
        images_batch, labels_batch = self.curr_queue.dequeue_many(self.batch_size_tf_var)
        return images_batch, labels_batch

    def __init__(self, train_features, train_labels, test_features, test_labels, batch_size):
        self.train_features = train_features[10000:]
        self.train_labels = train_labels[10000:]

        self.sesop_train_features = train_features[:10000]
        self.sesop_train_labels = train_labels[:10000]


        self.test_features = test_features
        self.test_labels = test_labels
        self.batch_size = batch_size


        self.batch_size_tf_var = tf.Variable(self.batch_size)
        self.batch_size_tf_placeholder = tf.placeholder(tf.int32)
        self.set_deque_batch_size_op = tf.assign(self.batch_size_tf_var, self.batch_size_tf_placeholder)

        self.dim = self.train_features.shape[1]

        self.label_dim = self.train_labels.shape[1]

        self.train_dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])
        self.train_dataY = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim])

        self.sesop_train_dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])
        self.sesop_train_dataY = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim])

        self.test_dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.dim])
        self.test_dataY = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim])

        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        # self.train_queue = tf.RandomShuffleQueue(shapes=[[self.dim], [self.label_dim]],
        #                                          dtypes=[tf.float32, tf.float32],
        #                                          capacity=6000,
        #                                          min_after_dequeue=1000,
        #                                          seed=1257812)

        self.train_queue = tf.FIFOQueue(shapes=[[self.dim], [self.label_dim]],
                                       dtypes=[tf.float32, tf.float32],
                                       capacity=50000)

        self.sesop_train_queue = tf.FIFOQueue(shapes=[[self.dim], [self.label_dim]],
                                        dtypes=[tf.float32, tf.float32],
                                        capacity=50000)

        self.test_queue = tf.FIFOQueue(shapes=[[self.dim], [self.label_dim]],
                                       dtypes=[tf.float32, tf.float32],
                                       capacity=50000)

        #0 mean test, 1 mean train
        self.data_source_idx = tf.Variable(tf.cast(1, tf.int32), trainable=False, name='data_source_idx')
        self.set_data_source_op = [tf.assign(self.data_source_idx, 0), tf.assign(self.data_source_idx, 1), tf.assign(self.data_source_idx, 2)]

        self.curr_queue = tf.QueueBase.from_list(tf.cast(self.data_source_idx, tf.int32),
                                       [self.test_queue, self.train_queue, self.sesop_train_queue])

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.train_enqueue_op = self.train_queue.enqueue_many([self.train_dataX, self.train_dataY])
        self.sesop_train_enqueue_op = self.sesop_train_queue.enqueue_many([self.sesop_train_dataX, self.sesop_train_dataY])
        self.test_enqueue_op = self.test_queue.enqueue_many([self.test_dataX, self.test_dataY])

        #self.train_generator_lock = threading.Lock()

    def sesop_train_data_iterator(self):
        """ A simple data iterator """
        batch_idx = 0
        while True:
            # shuffle labels and features
            idxs = np.arange(0, len(self.sesop_train_features))
            np.random.shuffle(idxs)
            shuf_features = self.sesop_train_features[idxs]
            shuf_labels = self.sesop_train_labels[idxs]
            for batch_idx in range(0, len(self.sesop_train_features), self.batch_size):
                images_batch = shuf_features[batch_idx:batch_idx + self.batch_size]
                images_batch = images_batch.astype("float32")
                labels_batch = shuf_labels[batch_idx:batch_idx + self.batch_size]
                yield images_batch, labels_batch


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

    def sesop_train_thread_main(self, sess):
        for dataX, dataY in self.sesop_train_data_iterator():
            sess.run(self.sesop_train_enqueue_op, feed_dict={self.sesop_train_dataX : dataX, self.sesop_train_dataY : dataY})

    def train_thread_main(self, sess):
        for dataX, dataY in self.train_data_iterator():
            sess.run(self.train_enqueue_op, feed_dict={self.train_dataX : dataX, self.train_dataY : dataY})

    def test_thread_main(self, sess):
        for dataX, dataY in self.test_data_iterator():
            sess.run(self.test_enqueue_op, feed_dict={self.test_dataX : dataX, self.test_dataY : dataY})

    def start_threads(self, sess, n_train_threads=1, n_test_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_train_threads):
            t = threading.Thread(target=self.train_thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)

        for n in range(1):
            t = threading.Thread(target=self.sesop_train_thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)

        for n in range(n_test_threads):
            t = threading.Thread(target=self.test_thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
