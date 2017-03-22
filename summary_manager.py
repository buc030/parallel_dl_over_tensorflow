
import tensorflow as tf
from threading import current_thread

class SummaryManager:
    insts = {}

    @classmethod
    def get(self):
        if current_thread() not in SummaryManager.insts:
            SummaryManager.insts[current_thread()] = SummaryManager()
        return SummaryManager.insts[current_thread()]


    def __init__(self):
        self.iter_summaries = []

    def add_iter_summary(self, s):
        self.iter_summaries.append(s)

    def merge_iters(self):
        return tf.summary.merge(self.iter_summaries)
