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


def build_train_example_queue(image, image_size, max_batch_size, depth):

    image = tf.image.resize_image_with_crop_or_pad(
            image, image_size + 4, image_size + 4)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.RandomShuffleQueue(
            capacity=32 * max_batch_size,
            min_after_dequeue=1 * max_batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
    return image, example_queue

def build_test_example_queue(image, image_size, max_batch_size, depth):

    image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.FIFOQueue(
            16 * max_batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
    return image, example_queue

def build_example_queue(image, label, image_size, max_batch_size, depth, is_training, num_threads):
    train_image, train_example_queue = build_test_example_queue(image, image_size, max_batch_size, depth)
    test_image, test_example_queue = build_train_example_queue(image, image_size, max_batch_size, depth)

    queue = tf.QueueBase.from_list(tf.cast(tf.equal(is_training, True), tf.int32), [test_example_queue, train_example_queue])
    image = tf.cond(tf.equal(is_training, True), lambda: train_image, lambda: test_image)

    train_example_enqueue_op = train_example_queue.enqueue([image, label])
    test_example_enqueue_op = test_example_queue.enqueue([image, label])

    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        queue, ([train_example_enqueue_op]* num_threads) + [test_example_enqueue_op]) )

    return image, queue


def build_input(dataset, data_path, test_path, batch_size, max_batch_size, is_training, num_threads):
  """Build CIFAR image and labels.

  Args:
    dataset: Either 'cifar10' or 'cifar100'.
    data_path: Filename for data.
    batch_size: Input batch size.
    mode: Either 'train' or 'eval'.
  Returns:
    images: Batches of images. [batch_size, image_size, image_size, 3]
    labels: Batches of labels. [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  enqueue_ops = []
  image_size = 32
  if dataset == 'cifar10':
    label_bytes = 1
    label_offset = 0
    num_classes = 10
  elif dataset == 'cifar100':
    label_bytes = 1
    label_offset = 1
    num_classes = 100
  else:
    raise ValueError('Not supported dataset %s', dataset)

  depth = 3
  image_bytes = image_size * image_size * depth
  record_bytes = label_bytes + label_offset + image_bytes

  data_files = tf.gfile.Glob(data_path)
  test_files = tf.gfile.Glob(test_path)
  #data_files = tf.gfile.Glob(path)

  #data_files = tf.gfile.Glob(data_path)
  #print ''
  train_file_queue = tf.train.string_input_producer(data_files, shuffle=True)
  test_file_queue = tf.train.string_input_producer(test_files, shuffle=False)

  file_queue = tf.QueueBase.from_list(tf.cast(tf.equal(is_training, True), tf.int32), [test_file_queue, train_file_queue])

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

  image, example_queue = build_example_queue(image, label, image_size, max_batch_size, depth, is_training, num_threads)

  # Read 'batch' labels + images from the example queue.
  images, labels = example_queue.dequeue_many(batch_size)
  labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  labels = tf.sparse_to_dense(
      tf.concat(values=[indices, labels], axis=1),
      [batch_size, num_classes], 1.0, 0.0)

  assert len(images.get_shape()) == 4

  assert images.get_shape()[-1] == 3
  assert len(labels.get_shape()) == 2

  assert labels.get_shape()[1] == num_classes

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, labels
