import tensorflow as tf
import numpy as np
import os


class ModelBuilder(object):

    def get_lr(self):
        assert (False)

    def get_input_shape(self):
        assert (False)

    def get_output_shape(self):
        assert (False)

    def get_checkpoint_name(self):
        assert (False)

    # return loss, accuracy, y_pred, train_step
    def build(self, x, y_, extra_params):
        assert (False)


class DebugRunner:

    def __init__(self, model_builder, datasets):
        self.nb_layers = []
        self.is_bn_on = tf.Variable(True, trainable=False)
        self.set_is_bn_on = [tf.assign(self.is_bn_on, False), tf.assign(self.is_bn_on, True)]

        self.datasets = datasets
        self.model_builder = model_builder


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

        for i in range(self.datasets.num_examples()/batch_size):
            batch_xs, batch_ys = self.datasets.train.next_batch(batch_size)
            cv_fd = {x: batch_xs, y_: batch_ys, keep_prob: 1.0,
                     phase_train: False}
            train_loss += loss.eval(cv_fd)
            train_accuracy += accuracy.eval(cv_fd)

        return train_accuracy/(self.datasets.num_examples()/batch_size), train_loss/(self.datasets.num_examples()/batch_size)


    def run(self):
        TASK = 'train'    # 'train' or 'test'
        chkpt_file = 'checkpoints/' + self.model_builder.get_checkpoint_name() + '.ckpt'

        # Train
        self.lr = tf.Variable(self.model_builder.get_lr(), name='learning_rate', trainable=False)
        self.mul_lr = tf.assign(self.lr, self.lr * 100)
        self.div_lr = tf.assign(self.lr, self.lr / 100)

        # Variables
        x = tf.placeholder(tf.float32, [None] + self.model_builder.get_input_shape(), name='x')
        y_ = tf.placeholder(tf.float32, [None] + self.model_builder.get_output_shape())

        keep_prob = tf.placeholder(tf.float32)
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        loss, accuracy, y_pred, train_step = self.model_builder.build(x, y_, keep_prob, phase_train)

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
                    for i in range(self.datasets.train.num_examples()/batch_size):
                        batch_xs, batch_ys = self.datasets.train.next_batch(batch_size)
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
            test_fd = {x: self.datasets.test.images, y_: self.datasets.test.labels,
                    keep_prob: 1.0, phase_train: False}
            print(' accuracy = %8.4f' % accuracy.eval(test_fd))
            # Multiclass Log Loss
            pred = y_pred.eval(test_fd)
            act = self.datasets.test.labels
            print(' multiclass logloss = %8.4f' % self.mlogloss(pred, act))

            # Save the variables to disk.
            if TASK == 'train':
                save_path = saver.save(sess, chkpt_file)
                print("Model saved in file: %s" % save_path)
