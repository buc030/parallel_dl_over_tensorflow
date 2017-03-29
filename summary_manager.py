
import tensorflow as tf

class SummaryManager:
    def __init__(self):
        self.iter_summaries = []

    def add_iter_summary(self, s):
        self.iter_summaries.append(s)

    def merge_iters(self):
        return tf.summary.merge(self.iter_summaries)

    def reset(self):
        self.iter_summaries = []
