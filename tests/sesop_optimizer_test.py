
import tensorflow as tf
import numpy as np
import shutil
import tf_utils
from tensorflow.examples.tutorials.mnist import input_data

from dataset_manager import DatasetManager
from batch_provider import SimpleBatchProvider

from seboost_optimizer import SeboostOptimizer


class FCLayer:
    def __init__(self, input, n_in, n_out, prefix, activation=True):
        with tf.name_scope(prefix):

            low = -np.sqrt(6.0 / (n_in + n_out))  # use 4 for sigmoid, 1 for tanh activation
            high = np.sqrt(6.0 / (n_in + n_out))

            #print 'prefix = ' + str(prefix)
            self.W = tf.Variable(tf.random_uniform([n_in, n_out], minval=low, maxval=high), dtype=tf.float32)
            self.b = tf.Variable(tf.zeros([n_out]))
            a = tf.matmul(input, self.W) + self.b

            #a = tf.layers.batch_normalization(a, trainable=False)

            if activation == False:
                self.out = a
            else:
                self.out = tf.nn.tanh(a)


class MLP:
    #labels,x are placeholders for the input and labels
    def __init__(self, x, labels, n_hidden, dim):


        layers = [FCLayer(x, dim, dim, 'FC', True)]
        for i in range(n_hidden - 1):
            layers.append(FCLayer(layers[-1].out, dim, dim, 'FC', True))

        layers.append(FCLayer(layers[-1].out, dim, 1, 'FC', True))

        self.layers = layers
        self.out = layers[-1].out

        # self.cross_entropy = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.out))
        # self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(labels, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        loss_per_sample = tf.squared_difference(self.out, labels, name='loss_per_sample')
        self.loss = tf.reduce_mean(loss_per_sample, name='loss')





bp = SimpleBatchProvider(input_dim=3, output_dim=1, dataset_size=50000, batch_size=100)
x, labels = bp.batch()

with tf.name_scope('main_model'):
    mlp = MLP(x, labels, 10, 3)

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Run a simple experiment.')
parser.add_argument('-lr', type=float, nargs=1, required=True, help='Starting learning rate')
parser.add_argument('-VECTOR_BREAKING', type=str2bool, nargs=1, required=True, help='')
parser.add_argument('-adaptable_learning_rate', type=str2bool, nargs=1, required=True, help='')

args = parser.parse_args()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = SeboostOptimizer(mlp.loss, bp, tf.trainable_variables(), history_size=10,
                                 VECTOR_BREAKING=args.VECTOR_BREAKING[0],
                                 lr=args.lr[0],
                                 batch_size=100,
                                 train_dataset_size=50000,
                                 adaptable_learning_rate=args.adaptable_learning_rate[0],
                                 num_of_batches_per_sesop=10)


with tf.Session() as sess:

    #shutil.rmtree('/tmp/generated_data/1')
    tensorboard_dir = tf_utils.allocate_tensorboard_dir()

    print 'Write graph into tensorboard into: ' + str(tensorboard_dir)
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(sess.graph)
    writer.flush()

    writer.add_summary(sess.run(tf.summary.text('args', tf.constant(str(args)))))
    optimizer.set_summary_writer(writer)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    bp.custom_runner.start_threads(sess)

    for epoch in range(100):
        optimizer.run_epoch(sess)

