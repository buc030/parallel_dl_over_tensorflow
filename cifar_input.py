# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CIFAR dataset input module.
"""

import tensorflow as tf


class CifarInput:
    def __init__(self, batch_size, path='./'):
        self.sesop_data_path = path + 'CIFAR_data/cifar-10-batches-bin/sesop_data_batch.bin'
        self.data_path = path + 'CIFAR_data/cifar-10-batches-bin/data_batch*'
        self.test_path = path + 'CIFAR_data/cifar-10-batches-bin/test_batch.bin'

        self.image_size = 32
        self.depth = 3
        self.num_classes = 10

        self.batch_size = batch_size
        self.batch_size_tf_var = tf.Variable(batch_size, trainable=False, name='batch_size_chooser')
        self.batch_size_tf_placeholder = tf.placeholder(tf.int32)
        self.set_deque_batch_size_op = tf.assign(self.batch_size_tf_var, self.batch_size_tf_placeholder)

        self.build_input()

    def get_inputs(self):
        with tf.name_scope('fetch_batch_from_queue'):
            images, labels = self.curr_queue.dequeue_many(self.batch_size_tf_var)

            # images = tf.Print(images, [tf.shape(images)])
            labels = tf.reshape(labels, [self.batch_size_tf_var, 1])
            indices = tf.reshape(tf.range(0, self.batch_size_tf_var, 1), [self.batch_size_tf_var, 1])
            labels = tf.sparse_to_dense(
                tf.concat(values=[indices, labels], axis=1),
                [self.batch_size_tf_var, self.num_classes], 1.0, 0.0)

            assert len(images.get_shape()) == 4

            assert images.get_shape()[-1] == 3
            assert len(labels.get_shape()) == 2

            assert labels.get_shape()[1] == self.num_classes

            # Display the training images in the visualizer.
            #tf.summary.image('images', images, max_outputs=3)
            return images, labels

    def set_data_source(self, sess, data_source_idx):
        sess.run(self.set_data_source_op[data_source_idx])

    def set_deque_batch_size(self, sess, new_batch_size):
        sess.run(self.set_deque_batch_size_op, feed_dict={self.batch_size_tf_placeholder: new_batch_size})


    ########### PRIVATE ##############
    def build_input(self):

        with tf.name_scope('test_data'):
            self.test_queue = self.build_queue(data_path=self.test_path, preprocess=False, shuffle=False, n_threads=1)

        with tf.name_scope('train_data'):
            self.train_queue = self.build_queue(data_path=self.data_path, preprocess=True, shuffle=True, n_threads=1)
            self.train_queue_size = self.train_queue.size()

        # with tf.name_scope('sesop_train_data'):
        #     self.sesop_train_queue = self.build_queue(data_path=self.sesop_data_path, preprocess=True, shuffle=True, n_threads=1)

        with tf.name_scope('choose_data_source'):
            self.data_source_idx = tf.Variable(tf.cast(1, tf.int32), trainable=False, name='data_source_idx')
            self.set_data_source_op = [tf.assign(self.data_source_idx, 0), tf.assign(self.data_source_idx, 1),
                                       tf.assign(self.data_source_idx, 2)]

            self.curr_queue = tf.QueueBase.from_list(tf.cast(self.data_source_idx, tf.int32),
                                                     [self.test_queue, self.train_queue])


    # build a queue that output examples from data_path
    # return the queue and the enqueu op for that queue
    def build_queue(self, data_path, preprocess=False, shuffle=False, n_threads=1):

        with tf.name_scope('read_image'):
            image, label = self.read_image(data_path)

        if preprocess == True:
            with tf.name_scope('preprocess'):
                image = self.preprocess_image(image)
        else:
            #anyways we need to do at least this:
            with tf.name_scope('resize_image_with_crop_or_pad'):
                image = tf.image.resize_image_with_crop_or_pad(image, self.image_size, self.image_size)
            with tf.name_scope('per_image_standardization'):
                image = tf.image.per_image_standardization(image)


        if shuffle == True:
            queue = tf.RandomShuffleQueue(shapes=[[self.image_size, self.image_size, self.depth], [1]],
                                          dtypes=[tf.float32, tf.int32],
                                          min_after_dequeue=50000,
                                          capacity=100000)
        else:
            queue = tf.FIFOQueue(shapes=[[self.image_size, self.image_size, self.depth], [1]],
                                 dtypes=[tf.float32, tf.int32],
                                 capacity=100000)

        with tf.name_scope('enqueue_op'):
            enqueue_op = queue.enqueue([image, label])
            qr = tf.train.QueueRunner(queue, enqueue_ops=[enqueue_op]*n_threads)
            tf.train.add_queue_runner(qr)

        return queue


    #create a graph that reads an image and a label from data_path
    def read_image(self, data_path):
        depth = 3
        label_bytes = 1
        label_offset = 0
        num_classes = 10
        image_size = 32

        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes

        # data_files = tf.gfile.Glob(data_path)
        files = tf.gfile.Glob(data_path)
        file_queue = tf.train.string_input_producer(files)


        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)

        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        # Convert from string to [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                                 [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        return image, label

    # create a graph that preprocess an image (resize, crop, flip)
    def preprocess_image(self, image):
        #hflip, random crop

        with tf.name_scope('resize_image_with_crop_or_pad'):
            image = tf.image.resize_image_with_crop_or_pad(image, self.image_size + 4, self.image_size + 4)

        with tf.name_scope('random_crop'):
            image = tf.random_crop(image, [self.image_size, self.image_size, self.depth])
        with tf.name_scope('random_flip_left_right'):
            image = tf.image.random_flip_left_right(image)


        with tf.name_scope('per_image_standardization'):
            image = tf.image.per_image_standardization(image)

        return image






