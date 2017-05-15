
import tensorflow as tf

from dataset_manager import DatasetManager
import cifar_input
from tf_utils import CustomRunner

class SimpleBatchProvider:
    def __init__(self, input_dim, output_dim, dataset_size, batch_size):
        self.batch_size = batch_size
        #return the same random data always.
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            DatasetManager().get_random_data(input_dim=input_dim, output_dim=output_dim, n=dataset_size)

        self.custom_runner = CustomRunner(self.training_data, self.training_labels,
                                          self.testing_data, self.testing_labels, batch_size)

        self._batch = self.custom_runner.get_inputs()

    def set_deque_batch_size(self, sess, new_batch_size):
        self.custom_runner.set_deque_batch_size(sess, new_batch_size)

    def set_data_source(self, sess, data_name='train'):
        if data_name == 'test':
            self.custom_runner.set_data_source(sess, 0)
        elif data_name == 'train':
            self.custom_runner.set_data_source(sess, 1)
        else:
            assert (data_name == 'sesop')
            self.custom_runner.set_data_source(sess, 2)

    def batch(self):
        return self._batch

class CifarBatchProvider:
    def __init__(self, batch_sizes):

        with tf.device('/cpu:*'):
            self.cifar_in = cifar_input.CifarInput(batch_sizes[0])

        self._batch = self.cifar_in.get_inputs()

    def set_deque_batch_size(self, sess, new_batch_size):
        self.cifar_in.set_deque_batch_size(sess, new_batch_size)

    def set_data_source(self, sess, data_name='train'):
        if data_name == 'test':
            self.cifar_in.set_data_source(sess, 0)
        elif data_name == 'train':
            self.cifar_in.set_data_source(sess, 1)
        else:
            assert (data_name == 'sesop')
            self.cifar_in.set_data_source(sess, 2)

    def batch(self):
        return self._batch

