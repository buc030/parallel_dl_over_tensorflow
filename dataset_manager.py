import tensorflow as tf
import numpy as np
import pickle
import os

def fc_layer(input, n_in, n_out):
    with tf.name_scope('FC'):
        W = tf.Variable(tf.random_normal([n_in, n_out]), name='W')
        b = tf.Variable(tf.zeros([n_out]), name='b')
        a = tf.matmul(input, W) + b

    out = tf.nn.tanh(a)


    return out


def build_model(x, y, dim):
    layers = [fc_layer(x, dim, 2*dim)]
    for i in range(1):
        layers.append(fc_layer(layers[-1], 2*dim, dim))
    layers.append(fc_layer(layers[-1], dim, 1))
    model_out = layers[-1]

    return model_out

def generate_random_data(dim, n):
    cov = np.random.rand(dim, dim)
    cov = np.dot(cov, cov.transpose())

    training_data = np.random.multivariate_normal(np.zeros(dim), cov, n)
    testing_data = np.random.multivariate_normal(np.zeros(dim), cov, n)

    with tf.name_scope('generating_data'):
        x = tf.placeholder(tf.float32, shape=[None, dim], name='x')
        model_out = build_model(x, None, dim)

        #with tf.Session('grpc://' + tf_server, config=config) as sess:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            training_labels = sess.run(model_out, feed_dict={x: training_data})
            testing_labels = sess.run(model_out, feed_dict={x: testing_data})

        return training_data, testing_data, training_labels, testing_labels


class DatasetManager:
    BASE_PATH = '/tmp/generated_data/DatasetManager/'

    def __init__(self):
        #make sure BASE_PATH exists
        if not os.path.exists(os.path.dirname(DatasetManager.BASE_PATH)):
            try:
                os.makedirs(os.path.dirname(DatasetManager.BASE_PATH))
            except os.OSError as exc:  # Guard against race condition
                if exc.errno != os.errno.EEXIST:
                    raise
    def get_random_data(self, dim, n):
        #if data is already there simply take it
        try:
            with open(DatasetManager.BASE_PATH + 'random_' + str(n) + '_' + str(dim), 'rb') as f:
                return pickle.load(f)
        except:
            pass

        #otherwise take it and dump it for next times
        data = generate_random_data(dim, n)
        print 'data shape = ' + str(data[0].shape) + ', dim = ' + str(dim)
        with open(DatasetManager.BASE_PATH + 'random_' + str(n) + '_' + str(dim), 'wb') as f:
            self.metadata = pickle.dump(data, f)

        return data