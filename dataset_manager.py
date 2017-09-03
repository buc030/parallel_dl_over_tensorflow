import tensorflow as tf
import numpy as np
import pickle
import os

def fc_layer(input, n_in, n_out, activation=True):
    with tf.name_scope('FC'):
        low = -np.sqrt(6.0 / (n_in + n_out))  # use 4 for sigmoid, 1 for tanh activation
        high = np.sqrt(6.0 / (n_in + n_out))

        # print 'prefix = ' + str(prefix)
        W = tf.Variable(tf.random_uniform([n_in, n_out], minval=low, maxval=high, dtype=tf.float32), name='W')
        b = tf.Variable(tf.zeros([n_out]), name='b')
        a = tf.matmul(input, W) + b

    if activation == False:
        return a

    out = tf.nn.tanh(a)


    return out


def build_model(x, dim, out_dim, layers_num=4):
    layers = [fc_layer(x, dim, dim/2)]
    for i in range(layers_num):
        layers.append(fc_layer(layers[-1], dim/2, dim/2))
    layers.append(fc_layer(layers[-1], dim/2, out_dim, False))
    model_out = layers[-1]

    return model_out

#def zero_random_matrix_value(cov):

def generate_random_data(input_dim, output_dim, n):
    cov = np.random.rand(input_dim, input_dim)
    cov = np.dot(cov, cov.transpose())

    #Make the problem harder:
    # for i in range(0):
    #     i = np.random.randint(input_dim)
    #     j = i
    #     while j == i:
    #         j = np.random.randint(input_dim)
    #
    #     cov[i][j] = 0

    # training_data = np.random.multivariate_normal(np.zeros(input_dim), cov, n)
    # testing_data = np.random.multivariate_normal(np.zeros(input_dim), cov, n)

    training_data = np.random.randn(n, input_dim)
    testing_data = np.random.randn(n, input_dim)

    with tf.name_scope('generating_data'):
        x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
        data_model_out = build_model(x, input_dim, input_dim, 1) #the data model is not deep, to have pretty scattered data
        label_model_out = build_model(x, input_dim, output_dim, 1)

        #with tf.Session('grpc://' + tf_server, config=config) as sess:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            training_data = sess.run(data_model_out, feed_dict={x: training_data})
            testing_data = sess.run(data_model_out, feed_dict={x: testing_data})
            # normalize the data:
            training_data = ((training_data - np.mean(training_data, 0)) / np.std(training_data, 0))
            testing_data = ((testing_data - np.mean(testing_data, 0)) / np.std(testing_data, 0))

            training_labels = sess.run(label_model_out, feed_dict={x: training_data})
            testing_labels = sess.run(label_model_out, feed_dict={x: testing_data})

        return training_data, testing_data, training_labels, testing_labels


class DatasetManager:
    BASE_PATH = '/home/shai/DatasetManager/'

    def __init__(self):
        #make sure BASE_PATH exists
        if not os.path.exists(os.path.dirname(DatasetManager.BASE_PATH)):
            try:
                os.makedirs(os.path.dirname(DatasetManager.BASE_PATH))
            except os.OSError as exc:  # Guard against race condition
                if exc.errno != os.errno.EEXIST:
                    raise
    def get_random_data(self, input_dim, output_dim, n):
        #if data is already there simply take it

        try:
            with open(DatasetManager.BASE_PATH + 'random_' + str(n) + '_inputdim_' + str(input_dim) + '_outputdim_' + str(output_dim), 'rb') as f:
                return pickle.load(f)
        except:
            pass

        #otherwise take it and dump it for next times
        data = generate_random_data(input_dim, output_dim, n)
        #print 'data shape = ' + str(data[0].shape) + ', dim = ' + str(dim)
        with open(DatasetManager.BASE_PATH + 'random_' + str(n) + '_inputdim_' + str(input_dim) + '_outputdim_' + str(output_dim), 'wb') as f:
            self.metadata = pickle.dump(data, f)

        return data