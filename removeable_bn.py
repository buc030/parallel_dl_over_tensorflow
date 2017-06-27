
import tensorflow as tf
import numpy as np

import os
from tensorflow.examples.tutorials.mnist import input_data

class RemoveableBNLayer:

    def __init__(self, _x, prev_layer_w, prev_layer_b, phase_train, is_bn_on):
        self.x = _x
        self.is_bn_on = is_bn_on
        #self.x = tf.Print(_x, [prev_layer_b])
        self.eps = 0.000
        params_shape = [self.x.get_shape()[-1]]

        self.beta = tf.Variable(
            name='beta', dtype=tf.float32,
            initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=True)

        self.gamma = tf.Variable(
            name='gamma', dtype=tf.float32,
            initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=True)

        #fully connected
        #self.bn_batch_mean, self.bn_batch_variance = tf.nn.moments(self.x, [0], name='moments')

        #conv: BHWD (batch, h, w, depth)
        self.bn_batch_mean, self.bn_batch_variance = tf.nn.moments(self.x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        # update moving avarge
        def mean_var_with_update():
            ema_apply_op = ema.apply([self.bn_batch_mean, self.bn_batch_variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(self.bn_batch_mean), tf.identity(self.bn_batch_variance)
                #return tf.Print(tf.identity(self.bn_batch_mean), [self.bn_batch_mean], 'update mean....'), tf.identity(self.bn_batch_variance)

        # update moving avarge
        def moving_mean_var():
            return (ema.average(self.bn_batch_mean), ema.average(self.bn_batch_variance))

        #dont update moving mean while switching BN/NON-BN
        self.is_not_updating = tf.Variable(True, trainable=False)
        self.set_is_not_updating = [tf.assign(self.is_not_updating, False), tf.assign(self.is_not_updating, True)]
        #SV NOTE: for now we only support switching using moving mean.
        self.mean, self.var = tf.cond(tf.logical_and(tf.logical_and(phase_train, self.is_bn_on), self.is_not_updating),
                            mean_var_with_update,
                            moving_mean_var)
                            #lambda: (ema.average(self.bn_batch_mean), ema.average(self.bn_batch_variance)))


        self.y_with_bn = tf.nn.batch_normalization(self.x, self.mean, self.var, self.beta, self.gamma, self.eps)
        self.y_without_bn = tf.identity(self.x)

        self.out = tf.cond(self.is_bn_on, lambda: self.y_with_bn, lambda: self.y_without_bn)


        ################# SAVE values when transforming from BN to non BN ##############
        self.saved_var = tf.Variable(
            name='saved_var', dtype=tf.float32,
            initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=False)

        self.saved_mean = tf.Variable(
            name='saved_mean', dtype=tf.float32,
            initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=False)

        self.saved_gamma = tf.Variable(
            name='saved_gamma', dtype=tf.float32,
            initial_value=tf.constant(1.0, tf.float32, params_shape), trainable=False)

        self.saved_beta = tf.Variable(
            name='saved_beta', dtype=tf.float32,
            initial_value=tf.constant(0.0, tf.float32, params_shape), trainable=False)


        tf.summary.histogram('W', prev_layer_w)
        tf.summary.histogram('b', prev_layer_b)

        tf.summary.histogram('gamma', self.gamma)
        tf.summary.histogram('beta', self.beta)

        tf.summary.histogram('var', self.var)
        tf.summary.histogram('mean', self.mean)

        tf.summary.histogram('saved_var', self.saved_var)
        tf.summary.histogram('saved_mean', self.saved_mean)

        tf.summary.histogram('saved_gamma', self.saved_gamma)
        tf.summary.histogram('saved_beta', self.saved_beta)

        self.save_ops = [tf.assign(self.saved_var, self.var), tf.assign(self.saved_mean, self.mean),
                         tf.assign(self.saved_beta, self.beta), tf.assign(self.saved_gamma, self.gamma)]

        ################# DEFINE transformation ###########################
        # (tf.sqrt(self.var) + self.eps)
        self.bn_to_nobn = [tf.assign(prev_layer_w, (self.gamma / (tf.sqrt(self.var) + self.eps)) * prev_layer_w),
                           tf.assign(prev_layer_b,
                                     (self.gamma / (tf.sqrt(self.var) + self.eps)) * (prev_layer_b - self.mean) + self.beta)]


        # W = W, b = b, gamma=sigma, mean=mu
        self.nobn_to_bn1 = [tf.assign(prev_layer_w, (tf.sqrt(self.saved_var)/ (self.saved_gamma + self.eps)) * prev_layer_w),
                           tf.assign(prev_layer_b,
                                     (tf.sqrt(self.saved_var) / (self.saved_gamma + self.eps)) * (prev_layer_b - self.saved_beta) + self.saved_mean)]

        self.nobn_to_bn2 = [tf.assign(self.gamma, (tf.sqrt(self.var)*self.saved_gamma)/(tf.sqrt(self.saved_var) + self.eps)),
                           tf.assign(self.beta, self.saved_beta + (self.saved_gamma/(tf.sqrt(self.saved_var) + self.eps))*(self.mean - self.saved_mean))]

    def pop_bn(self, sess, fd):
        sess.run(self.set_is_not_updating[0])
        sess.run(self.save_ops, feed_dict=fd)
        sess.run(self.bn_to_nobn, feed_dict=fd)
        sess.run(self.set_is_not_updating[1])

    def push_bn(self, sess, fd):
        sess.run(self.set_is_not_updating[0])
        sess.run(self.nobn_to_bn1, feed_dict=fd)
        sess.run(self.nobn_to_bn2, feed_dict=fd)
        sess.run(self.set_is_not_updating[1])

# Max Pooling Layer
class MaxPooling2D(object):
    '''
      constructor's args:
          input  : input image (2D matrix)
          ksize  : pooling patch size
    '''
    def __init__(self, input, ksize=None):
        self.input = input
        if ksize == None:
            ksize = [1, 2, 2, 1]
            self.ksize = ksize

    def output(self):
        self.output = tf.nn.max_pool(self.input, ksize=self.ksize,
                    strides=[1, 2, 2, 1], padding='SAME')

        return self.output

class Convolution2D(object):
    '''
      constructor's args:
          input     : input image (2D matrix)
          input_siz ; input image size
          in_ch     : number of incoming image channel
          out_ch    : number of outgoing image channel
          patch_siz : filter(patch) size
          weights   : (if input) (weights, bias)
    '''
    def __init__(self, input, input_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input
        self.rows = input_siz[0]
        self.cols = input_siz[1]
        self.in_ch = in_ch
        self.activation = activation

        wshape = [patch_siz[0], patch_siz[1], in_ch, out_ch]

        w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),
                            trainable=True, name='W')
        b_cv = tf.Variable(tf.constant(0.1, shape=[out_ch]),
                            trainable=True, name='b')

        self.w = w_cv
        self.b = b_cv
        self.params = [self.w, self.b]

    def output(self):
        shape4D = [-1, self.rows, self.cols, self.in_ch]

        x_image = tf.reshape(self.input, shape4D)  # reshape to 4D tensor
        linout = tf.nn.conv2d(x_image, self.w,
                  strides=[1, 1, 1, 1], padding='SAME') + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        else:
            self.output = linout

        return self.output


# Full-connected Layer
class FullConnected(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_h = tf.Variable(tf.truncated_normal([n_in,n_out],
                          mean=0.0, stddev=0.05), trainable=True)
        b_h = tf.Variable(tf.zeros([n_out]), trainable=True)

        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)

        return self.output


# Read-out Layer
class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_o = tf.Variable(tf.random_normal([n_in,n_out],
                        mean=0.0, stddev=0.05), trainable=True)
        b_o = tf.Variable(tf.zeros([n_out]), trainable=True)

        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.softmax(linarg)

        return self.output
#

class MnistModel:

    def pop_bns(self, sess, fd):
        for bn in self.nb_layers[::-1]:
            bn.pop_bn(sess, fd)

        sess.run(self.set_is_bn_on[0])

    def push_bns(self, sess, fd):
        for bn in self.nb_layers:
            bn.push_bn(sess, fd)

        sess.run(self.set_is_bn_on[1])

    def __init__(self):
        self.nb_layers = []
        self.is_bn_on = tf.Variable(True, trainable=False)
        self.set_is_bn_on = [tf.assign(self.is_bn_on, False), tf.assign(self.is_bn_on, True)]

        self.mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    def batch_norm(self, x, prev_layer_w, prev_layer_b, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):
            bn_layer = RemoveableBNLayer(_x=x, prev_layer_w=prev_layer_w, prev_layer_b=prev_layer_b, phase_train=phase_train, is_bn_on=self.is_bn_on)
            self.nb_layers.append(bn_layer)
            normed = bn_layer.out

        return normed

    def training(self, loss, learning_rate):
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def evaluation(self, y_pred, y):
        correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        return accuracy

    def inference(self, x, y_, keep_prob, phase_train):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        self.layers = []
        with tf.variable_scope('conv_1'):
            #input, input_siz, in_ch, out_ch, patch_siz, activation
            conv1 = Convolution2D(x, (28, 28), 1, 32, (5, 5), activation='none')
            self.layers.append(conv1)
            #conv1_bn = self.batch_norm(conv1.output(), 32, phase_train)
            conv1_bn = self.batch_norm(conv1.output(), conv1.w, conv1.b, phase_train)
            conv1_out = tf.nn.relu(conv1_bn)

            pool1 = MaxPooling2D(conv1_out)
            pool1_out = pool1.output()
            #pool1_flat1 = tf.reshape(pool1_out, [-1, 14 * 14 * 32])

        with tf.variable_scope('conv_2'):
            conv2 = Convolution2D(pool1_out, (28, 28), 32, 64, (5, 5),
                                  activation='none')
            self.layers.append(conv2)
            conv2_bn = self.batch_norm(conv2.output(), conv2.w, conv2.b, phase_train)
            conv2_out = tf.nn.relu(conv2_bn)
            pool2 = MaxPooling2D(conv2_out)
            pool2_out = pool2.output()
            pool2_flat = tf.reshape(pool2_out, [-1, 7 * 7 * 64])

        with tf.variable_scope('fc1'):
            fc1 = FullConnected(pool2_flat, 7 * 7 * 64, 1024)
            self.layers.append(fc1)
            fc1_out = fc1.output()
            fc1_dropped = fc1_out
            #fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)

        #just FC with softmax
        with tf.variable_scope('fc2'):
            y_pred = ReadOutLayer(fc1_dropped, 1024, 10).output()
            self.layers.append(y_pred)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_pred),
                                                      reduction_indices=[1]))

        loss = cross_entropy

        #add penalties
        # for bn in self.nb_layers:
        #     loss += bn.penelty_gamma
        #     loss += bn.penelty_beta

        train_step = self.training(loss, self.lr)
        accuracy = self.evaluation(y_pred, y_)

        return loss, accuracy, y_pred, train_step

    def mlogloss(self, predicted, actual):
        '''
          args.
             predicted : predicted probability
                        (sum of predicted proba should be 1.0)
             actual    : actual value, label
        '''

        def inner_fn(item):
            eps = 1.e-15
            item1 = min(item, (1 - eps))
            item1 = max(item, eps)
            res = np.log(item1)

            return res

        nrow = actual.shape[0]
        ncol = actual.shape[1]

        mysum = sum([actual[i, j] * inner_fn(predicted[i, j])
                     for i in range(nrow) for j in range(ncol)])

        ans = -1 * mysum / nrow

        return ans

    def calc_train_loss(self, x, y_, loss, accuracy, batch_size, phase_train, keep_prob):
        train_accuracy = 0.0
        train_loss = 0.0

        for i in range(55000/batch_size):
            batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
            cv_fd = {x: batch_xs, y_: batch_ys, keep_prob: 1.0,
                     phase_train: False}
            train_loss += loss.eval(cv_fd)
            train_accuracy += accuracy.eval(cv_fd)

        return train_accuracy/(55000/batch_size), train_loss/(55000/batch_size)


    def run(self):
        TASK = 'train'    # 'train' or 'test'
        chkpt_file = 'checkpoints/mnist_cnn.ckpt'

        # Train
        self.lr = tf.Variable(0.001, name='learning_rate', trainable=False)
        self.mul_lr = tf.assign(self.lr, self.lr * 100)
        self.div_lr = tf.assign(self.lr, self.lr / 100)

        # Variables
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        y_ = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        loss, accuracy, y_pred, train_step = self.inference(x, y_,
                                             keep_prob, phase_train)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        full_train_loss = tf.placeholder(tf.float32)
        full_train_accuracy = tf.placeholder(tf.float32)
        tf.summary.scalar('full_train_loss', full_train_loss, ['full_train_summaries'])
        tf.summary.scalar('full_train_accuracy', full_train_accuracy, ['full_train_summaries'])


        vars_to_train = tf.trainable_variables()    # option-1

        ############### BN LR experiment ################
        _grad = tf.gradients(loss, vars_to_train)
        grad_norms = [tf.norm(g) for g in _grad]

        global_grad_norm = tf.global_norm(_grad)
        snapshot = [tf.Variable(v.initialized_value(), trainable=False) for v in vars_to_train]
        #distances = [tf.squared_difference(v1, v2) for v1,v2 in zip(snapshot, vars_to_train)]
        self.take_snapshot = [tf.assign(v1, v2) for v1,v2 in zip(snapshot, vars_to_train)]
        [tf.summary.scalar('BN_effective_lr_' + str(v2.name), tf.norm(v1 - v2)/grad_norm , ['BN_effective_lr_summary']) for v1,v2, grad_norm in zip(snapshot, vars_to_train, grad_norms)]
        tf.summary.scalar('BN_effective_lr_global', tf.global_norm([v1 - v2 for v1,v2 in zip(snapshot, vars_to_train)])/global_grad_norm, ['BN_effective_lr_summary'])
        #################################################


        vars_for_bn1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, # TF >1.0
                                         scope='conv_1/bn')
        vars_for_bn2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, # TF >1.0
                                         scope='conv_2/bn')

        vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
        vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))

        if TASK == 'test' or os.path.exists(chkpt_file):
            restore_call = True
            vars_all = tf.all_variables()
            vars_to_init = list(set(vars_all) - set(vars_to_train))
            init = tf.variables_initializer(vars_to_init)   # TF >1.0
        elif TASK == 'train':
            restore_call = False
            init = tf.global_variables_initializer()    # TF >1.0
        else:
            print('Check task switch.')

        saver = tf.train.Saver(vars_to_train)     # option-1
        # saver = tf.train.Saver()                   # option-2

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)


        with tf.Session() as sess:
            # if TASK == 'train':              # add in option-2 case
            sess.run(init)                     # option-1

            if restore_call:
                # Restore variables from disk.
                saver.restore(sess, chkpt_file)

            full_train_merged = tf.summary.merge_all('full_train_summaries')
            merged = tf.summary.merge_all()
            BN_effective_lr_summary = tf.summary.merge_all('BN_effective_lr_summary')

            train_writer = tf.summary.FileWriter('./tensorboard', sess.graph)

            if TASK == 'train':
                batch_size = 100
                print('\n Training...')
                for epoch in range(200):
                    print ('epoch = %d', epoch)
                    #train
                    #SV DEBUG
                    for i in range(55000/batch_size):
                        batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                        #SV NOTE: dont update mocing mean when running without bn
                        #is_bn_on = epoch % 2 == 0
                        if i % 10 == 0:
                            print ('i = ' + str(i))
                        #Now add BN, and make a step
                        if epoch > 0 or i > 0:
                            _grad_norms = sess.run(grad_norms, {x: batch_xs, y_: batch_ys, phase_train: False})
                            _global_grad_norm = sess.run(global_grad_norm, {x: batch_xs, y_: batch_ys, phase_train: False})
                            sess.run(self.take_snapshot)
                            self.push_bns(sess, {x: batch_xs, phase_train: False, keep_prob: 1.0})

                        _ = sess.run(train_step, {x: batch_xs, y_: batch_ys, keep_prob: 0.5,
                              phase_train: True})

                        # train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.5,
                        #       phase_train: True})

                        #Now pop the BN and check how far did we go?
                        #This way we can learn how BN affect learning rates
                        if epoch > 0 or i > 0:
                            fd = {}
                            fd[global_grad_norm] = _global_grad_norm
                            fd.update({x: batch_xs, phase_train: False, keep_prob: 1.0})
                            for gn, _gn in zip(grad_norms, _grad_norms):
                                fd[gn] = _gn
                            self.pop_bns(sess, {x: batch_xs, phase_train: False, keep_prob: 1.0})
                            train_writer.add_summary(sess.run(BN_effective_lr_summary, fd), i + epoch * len(range(55000 / batch_size)))

                        if epoch > 0:
                            summary = sess.run(merged, {x: batch_xs, y_: batch_ys, phase_train: False, keep_prob: 1.0})
                            train_writer.add_summary(summary, i + epoch*len(range(55000/batch_size)))




                    train_accuracy, train_loss = self.calc_train_loss(x, y_, loss, accuracy, batch_size, phase_train, keep_prob)
                    print('  step, loss, accurary = %6d: %8.4f, %8.4f' % (epoch, train_loss, train_accuracy))

                    summary = sess.run(full_train_merged, {full_train_accuracy: train_accuracy, full_train_loss: train_loss})
                    train_writer.add_summary(summary, epoch)

                    # bn_switch_batch_x, bn_switch_batch_y = self.mnist.train.next_batch(batch_size * 10)
                    # if epoch % 2 == 0:
                    #     print ('poping bn...')
                    #     #self.pop_bns(sess, {x: bn_switch_batch_x, phase_train: True, keep_prob: 1.0})
                    #     #sess.run(self.div_lr)
                    #     #sess.run(merged)
                    # else:
                    #     print ('pushing bn...')
                    #     #self.push_bns(sess, {x: bn_switch_batch_x, phase_train: True, keep_prob: 1.0})
                    #     #sess.run(self.mul_lr)
                    #     #sess.run(merged)

                    # train_accuracy, train_loss = self.calc_train_loss(x, y_, loss, accuracy, batch_size, phase_train, keep_prob)
                    # print('(after) step, loss, accurary = %6d: %8.4f, %8.4f' % (epoch, train_loss, train_accuracy))
                    print ('-----------------------------------------------')
                    train_writer.flush()

            # Test trained model
            test_fd = {x: self.mnist.test.images, y_: self.mnist.test.labels,
                    keep_prob: 1.0, phase_train: False}
            print(' accuracy = %8.4f' % accuracy.eval(test_fd))
            # Multiclass Log Loss
            pred = y_pred.eval(test_fd)
            act = self.mnist.test.labels
            print(' multiclass logloss = %8.4f' % self.mlogloss(pred, act))

            # Save the variables to disk.
            if TASK == 'train':
                save_path = saver.save(sess, chkpt_file)
                print("Model saved in file: %s" % save_path)
#
model = MnistModel()
model.run()
