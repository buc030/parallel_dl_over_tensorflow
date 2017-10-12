
import tensorflow as tf

from dataset_manager import DatasetManager
import cifar_input
from tf_utils import CustomRunner
from tensorflow.examples.tutorials.mnist import input_data

class SimpleBatchProvider:

    def sesop_train_size(self):
        return self.sesop_dataset_size

    def train_size(self):
        return self.dataset_size - self.sesop_dataset_size

    def test_size(self):
        return self.dataset_size


    def __init__(self, input_dim, output_dim, dataset_size, batch_size, sesop_dataset_size):

        self.sesop_dataset_size = sesop_dataset_size
        self.dataset_size = dataset_size

        #return the same random data always.
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            DatasetManager().get_random_data(input_dim=input_dim, output_dim=output_dim, n=dataset_size)

        self.custom_runner = CustomRunner(self.training_data, self.training_labels,
                                          self.testing_data, self.testing_labels, batch_size, sesop_dataset_size)

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


class PhysicsBatchProvider:

    def sesop_train_size(self):
        return self.sesop_dataset_size

    def train_size(self):
        return self.train_dataset_size

    def test_size(self):
        return self.test_dataset_size


    def __init__(self, batch_size, test_dataset_size, sesop_dataset_size, seed):

        from numpy import genfromtxt
        data = genfromtxt('/home/shai/DatasetManager/michael_synthesised/xNrm.csv', delimiter=',')
        labels = genfromtxt('/home/shai/DatasetManager/michael_synthesised/y460.csv', delimiter=',')

        self.dataset_size = data.shape[0]

        self.test_dataset_size = test_dataset_size
        self.sesop_dataset_size = sesop_dataset_size
        self.train_dataset_size = self.dataset_size - self.test_dataset_size - self.sesop_dataset_size


        #return the same random data always.
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = data[:self.train_dataset_size + self.sesop_dataset_size], data[self.train_dataset_size + self.sesop_dataset_size:], \
                                                                                           labels[:self.train_dataset_size + self.sesop_dataset_size], labels[self.train_dataset_size + self.sesop_dataset_size:]

        self.custom_runner = CustomRunner(self.training_data, self.training_labels,
                                          self.testing_data, self.testing_labels, batch_size, sesop_dataset_size, seed)

        self._batch = self.custom_runner.get_inputs()

    def set_deque_batch_size(self, sess, new_batch_size):
        self.custom_runner.set_deque_batch_size(sess, new_batch_size)

    def set_data_source(self, sess, data_name='train'):
        if data_name == 'test':
            self.custom_runner.set_data_source(sess, 0)
        elif data_name == 'train':
            self.custom_runner.set_data_source(sess, 1)
        else:
            assert (data_name == 'sesop' and self.sesop_dataset_size > 0)
            self.custom_runner.set_data_source(sess, 2)

    def batch(self):
        return self._batch

import numpy

def mess_with_image(images, image_label, num_of_images):
    messed_with = []
    counter = 0
    for i in range(len(images[0])):
        pixels = images[0][i].reshape((28, 28))

        center = numpy.random.randint(2, 26)

        if images[1][i] == image_label:
            messed_with.append(i)
            #print type(pixels)
            #print pixels.shape
            pixels[center][center] = 1.0
            pixels[center][center + 1] = 1.0
            pixels[center][center - 1] = 1.0

            pixels[center + 1][center] = 1.0
            pixels[center - 1][center] = 1.0

            counter += 1
            if counter >= num_of_images:
                return messed_with


class FashionMnistBatchProvider:
    VALIDATION_SIZE = 5000

    def sesop_train_size(self):
        return self.sesop_dataset_size

    def train_size(self):
        return self.mnist.train.num_examples - self.sesop_dataset_size

    def test_size(self):
        return self.mnist.test.num_examples

    def __init__(self, initial_batch_size, sesop_private_dataset, seed, mess_with_data=0):

        self.sesop_private_dataset = sesop_private_dataset
        if sesop_private_dataset == True:
            self.sesop_dataset_size = 10000
        else:
            self.sesop_dataset_size = 0

        self.mnist = input_data.read_data_sets("/home/shai/tensorflow/fashion-mnist/data/fashion/", one_hot=False, validation_size=FashionMnistBatchProvider.VALIDATION_SIZE)
        train_data = self.mnist.train.next_batch(self.mnist.train.num_examples - FashionMnistBatchProvider.VALIDATION_SIZE)
        test_data = self.mnist.test.next_batch(self.mnist.test.num_examples)

        if mess_with_data > 0:
            #We had total dataset_size/10 bad examples.
            #and say we have mess_with_data of them in the train set. such that they are all from label 0.

            mess_with_image(train_data, 0, mess_with_data)

            #in the test set we have dataset_size/10 - mess_with_data bad examples.
            # (dataset_size/10)/10 - mess_with_data of label 1
            # (dataset_size/10)/10 of label 1
            # ...
            # (dataset_size/10)/10 of label 9
            mess_with_image(test_data, 0, 600 - mess_with_data)
            for i in range(9):
                mess_with_image(test_data, i + 1, 600)

        self.custom_runner = CustomRunner(train_data[0], train_data[1],
                                          test_data[0], test_data[1],
                                          initial_batch_size, self.sesop_dataset_size, seed)

        self._batch = self.custom_runner.get_inputs()


    def set_deque_batch_size(self, sess, new_batch_size):
        assert (new_batch_size < 50000)
        self.custom_runner.set_deque_batch_size(sess, new_batch_size)

    def set_data_source(self, sess, data_name='train'):
        if data_name == 'test':
            self.custom_runner.set_data_source(sess, 0)
        elif data_name == 'train':
            self.custom_runner.set_data_source(sess, 1)
        else:
            assert (data_name == 'sesop' and self.sesop_private_dataset == True)
            self.custom_runner.set_data_source(sess, 2)


    def batch(self):
        return self._batch




class MnistBatchProvider:
    VALIDATION_SIZE = 5000

    def sesop_train_size(self):
        return self.sesop_dataset_size

    def train_size(self):
        return self.mnist.train.num_examples - self.sesop_dataset_size

    def test_size(self):
        return self.mnist.test.num_examples

    def __init__(self, initial_batch_size, sesop_private_dataset, seed, mess_with_data = 0):

        self.sesop_private_dataset = sesop_private_dataset
        if sesop_private_dataset == True:
            self.sesop_dataset_size = 10000
        else:
            self.sesop_dataset_size = 0

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, validation_size=MnistBatchProvider.VALIDATION_SIZE)
        train_data = self.mnist.train.next_batch(self.mnist.train.num_examples - MnistBatchProvider.VALIDATION_SIZE)
        test_data = self.mnist.test.next_batch(self.mnist.test.num_examples)
        if mess_with_data > 0:
            mess_with_image(train_data, 1, mess_with_data)

        self.custom_runner = CustomRunner(train_data[0], train_data[1],
                                          test_data[0], test_data[1],
                                          initial_batch_size, self.sesop_dataset_size, seed)

        self._batch = self.custom_runner.get_inputs()


    def set_deque_batch_size(self, sess, new_batch_size):
        assert (new_batch_size < 50000)
        self.custom_runner.set_deque_batch_size(sess, new_batch_size)

    def set_data_source(self, sess, data_name='train'):
        if data_name == 'test':
            self.custom_runner.set_data_source(sess, 0)
        elif data_name == 'train':
            self.custom_runner.set_data_source(sess, 1)
        else:
            assert (data_name == 'sesop' and self.sesop_private_dataset == True)
            self.custom_runner.set_data_source(sess, 2)


    def batch(self):
        return self._batch



class CifarBatchProvider:
    def __init__(self, initial_batch_size, path='./'):

        with tf.device('/cpu:*'):
            self.cifar_in = cifar_input.CifarInput(initial_batch_size, path)

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

