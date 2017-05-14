
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

