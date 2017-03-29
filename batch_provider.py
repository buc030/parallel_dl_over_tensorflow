
import numpy
import tensorflow as tf


class BatchProvider:

    ##private
    def create_batching_pipeline(self, full_input, full_label):
        input, label = tf.train.slice_input_producer([full_input, full_label], name='slicer', shuffle=True, seed=1)
        batched_input, batched_labels = tf.train.batch([input, label], batch_size=self.curr_batch_size_tf, name='batcher', capacity=2*max(self.batch_sizes))

        return batched_input, batched_labels

    def build_output(self):
        cases = []
        for bs, batch in self.batch_pipes.items():

            f = lambda inst=self,bs=bs: inst.batch_pipes[bs]
            cases.append((tf.equal(self.curr_batch_size_tf, bs), f))

            #op = tf.cond(tf.equal(self.curr_batch_size_tf, bs), lambda: batch, lambda: op)

        #for c in cases:
        #    print 'c= ' + str(c[1]())
        #return op
        return tf.case(cases, default=lambda: self.batch_pipes.values()[-1], exclusive=True)


    ##public
    #User has to say infornt what batch sizes does he wants to support!
    def __init__(self, training_data, testing_data, training_labels, testing_labels, batch_sizes):
        with tf.name_scope('data_provider'):
            self.sub_init(training_data, testing_data, training_labels, testing_labels, batch_sizes)

    def sub_init(self, training_data, testing_data, training_labels, testing_labels, batch_sizes):
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            training_data, testing_data, training_labels, testing_labels

        self.batch_sizes = batch_sizes
        print 'data shape ' + str(training_data.shape)
        #set the provider in training mode.
        self.is_training = tf.Variable(True, dtype=tf.bool, name='is_training', trainable=False) #must be feed with dict_feed.
        self.curr_batch_size_tf = tf.Variable(batch_sizes[0], dtype=tf.int32, name='curr_batch_size', trainable=False)


        self.full_train_input  = tf.get_variable(name='train_dataset_x', initializer=lambda shape, dtype, partition_info:  training_data, shape=training_data.shape, trainable=False)
        self.full_train_labels = tf.get_variable(name='train_dataset_y', initializer=lambda shape, dtype, partition_info:  training_labels, shape=training_labels.shape, trainable=False)

        self.full_test_input  = tf.get_variable(name='test_dataset_x', initializer=lambda shape, dtype, partition_info:  testing_data, shape=testing_data.shape, trainable=False)
        self.full_test_labels = tf.get_variable(name='test_dataset_y', initializer=lambda shape, dtype, partition_info:  testing_labels, shape=testing_labels.shape, trainable=False)

        #Map from batch size to pipe that produce that pipe
        training_pipe = self.create_batching_pipeline(self.full_train_input, self.full_train_labels)
        testing_pipe = self.create_batching_pipeline(self.full_test_input, self.full_test_labels)
        batched_input, batched_labels = tf.case([(tf.equal(self.is_training, True), \
                                                lambda: training_pipe)], \
                                                default=lambda: testing_pipe)

        #batched_input, batched_labels
        """
        self.batch_pipes = {}
        training_pipe, testing_pipe = {}, {}

        for batch_size in batch_sizes:
            #build the pipelines
            print 'building batch ' + str(batch_size)
            with tf.name_scope('batch_size_' + str(batch_size)):
                training_pipe[batch_size] = self.create_batching_pipeline(self.full_train_input, self.full_train_labels, batch_size)
                testing_pipe[batch_size] = self.create_batching_pipeline(self.full_test_input, self.full_test_labels, batch_size)
                #build the case
                with tf.name_scope('choose_train_or_test'):
                    batched_input, batched_labels = tf.case([(tf.equal(self.is_training, True),\
                                lambda: training_pipe[batch_size])],\
                        default=lambda: testing_pipe[batch_size])

            self.batch_pipes[batch_size] = batched_input, batched_labels

        with tf.name_scope('choose_batch_size'):
            batched_input, batched_labels = self.build_output()
        """


        self.out_data = batched_input
        self.out_label = batched_labels

        self.set_training = {True : tf.assign(self.is_training, True), False : tf.assign(self.is_training, False)}

        self.set_batch_size = {}
        for b in batch_sizes:
            self.set_batch_size[b] = tf.assign(self.curr_batch_size_tf, b)

    def set_training_op(self, val):
        return self.set_training[val]

    def set_batch_size_op(self, batch_size):
        assert (batch_size in self.set_batch_size)
        return self.set_batch_size[batch_size]

    #this creates an op when its called with a new batch_size.
    def batch(self):
        return self.out_data, self.out_label


