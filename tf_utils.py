
import tensorflow as tf
import progressbar
from progressbar import ProgressBar, Percentage, Bar, ETA
from time import sleep





#returns an op that concat op_func() n times
#op_func is a function that creates an op
def chain_training_step(op_func, n):
    print 'Creating ' + str(n) + ' chained training steps'
    if n == 0:
        return tf.no_op()
    bar = progressbar.ProgressBar(maxval=n, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    res = tf.no_op()
    for i in range(n):
        with tf.control_dependencies([res]):
            step = op_func()
            res = tf.group(res, step)
        bar.update(i + 1)
    bar.finish()
    return res






class StackedBatches:
    def __init__(self):
        self.batches = []
        self.batch_sizes = []

    def add_batch(self, batch, batch_size):
        self.batches.append(batch)
        self.batch_sizes.append(batch_size)

    def build_stacked_batches(self):
        self._stack_batches = tf.concat(values=self.batches, axis=0)

        self.batch_sizes_2_indexes = {}
        for target_batch_size in self.batch_sizes:
            first_idx = 0
            for b in self.batch_sizes:
                if target_batch_size == b:
                    self.batch_sizes_2_indexes[target_batch_size] = first_idx
                    break
                first_idx += b
            assert(target_batch_size  in self.batch_sizes_2_indexes)

    def get_batch_by_batchsize_op(self, batch_size):
        index = self.batch_sizes_2_indexes[batch_size]
        return self._stack_batches[index : index+batch_size]
