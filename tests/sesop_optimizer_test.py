
import tensorflow as tf
import numpy as np
import shutil

from tensorflow.examples.tutorials.mnist import input_data

from dataset_manager import DatasetManager
from batch_provider import SimpleBatchProvider

from sesop_optimizer import SubspaceOptimizer


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



update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

lr = tf.Variable(0.02, dtype=tf.float32, trainable=False)
lr_placeholder = tf.placeholder(dtype=tf.float32)
update_lr = tf.assign(lr, lr_placeholder)

with tf.name_scope('sesop_optimizer'):
    with tf.control_dependencies(update_ops):
        sgd_optim = tf.train.GradientDescentOptimizer(lr)
        sgd_grads = sgd_optim.compute_gradients(mlp.loss)

        sgd_op = sgd_optim.minimize(mlp.loss)


full_loss_summary = tf.summary.scalar('full_loss', mlp.loss, ['full_loss'])
tf.summary.scalar('loss_during_sgd', mlp.loss, ['sgd_summaries'])
tf.summary.scalar('weights_norm_during_sgd', tf.global_norm(tf.trainable_variables()), ['sgd_summaries'])
tf.summary.scalar('grad_norm_norm_during_sgd', tf.global_norm(sgd_grads), ['sgd_summaries'])
tf.summary.scalar('learning_rate', lr, ['sgd_summaries'])


with tf.name_scope('sesop_optimizer'):
    with tf.control_dependencies(update_ops):
        optim = SubspaceOptimizer(mlp.loss, tf.trainable_variables(), 10, VECTOR_BREAKING=False)


with tf.Session() as sess:
    print 'Write graph into tensorboard'
    shutil.rmtree('/tmp/generated_data/1')
    writer = tf.summary.FileWriter('/tmp/generated_data/1')
    writer.add_graph(sess.graph)
    writer.flush()

    optim.set_summary_writer(writer)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    bp.custom_runner.start_threads(sess)

    sgd_summaries = tf.summary.merge_all('sgd_summaries')

    for i in range(100):
        #do 100 sgd steps
        for j in range(100):
            s = sess.run(sgd_summaries)
            writer.add_summary(s, i*100 + j)

            sess.run(sgd_op)




        feed_dicts = []
        for z in range(10):
            _x, _labels = sess.run([x, labels])
            feed_dicts.append({ x : _x, labels : _labels})


        bp.set_deque_batch_size(sess, 50000)
        writer.add_summary(sess.run(full_loss_summary), 2*i)
        bp.set_deque_batch_size(sess, 100)

        print 'i = ' + str(i)
        print 'loss before = ' + str(sess.run(mlp.loss, feed_dicts[0]))
        _distance_sesop_moved = optim.minimize(session=sess, feed_dicts=feed_dicts)
        if _distance_sesop_moved is not None:
            pass
            # print 'setting lr to: ' + str(_distance_sesop_moved/500.00)
            # sess.run(update_lr, {lr_placeholder : _distance_sesop_moved/500.00})

        print 'loss  after = ' + str(sess.run(mlp.loss, feed_dicts[0]))
        print '------------------'

        bp.set_deque_batch_size(sess, 50000)
        writer.add_summary(sess.run(full_loss_summary), 2*i + 1)
        bp.set_deque_batch_size(sess, 100)
