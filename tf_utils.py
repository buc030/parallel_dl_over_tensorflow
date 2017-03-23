
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
