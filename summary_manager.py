
import tensorflow as tf

class SummaryManager:
    def __init__(self, path):
        self.iter_summaries = []
        self.path = path
        self.writer = tf.summary.FileWriter(path)

    def add_iter_summary(self, s):
        self.iter_summaries.append(s)

    def merge_iters(self):
        #print 'self.iter_summaries = ' + str(self.iter_summaries)
        return tf.summary.merge(self.iter_summaries)

    def reset(self):
        self.iter_summaries = []
