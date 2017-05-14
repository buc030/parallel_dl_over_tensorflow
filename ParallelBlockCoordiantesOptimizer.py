
import tensorflow as tf
from VarOptimizer import VarOptimizer
import threading
import numpy as np

class ParallelBlockCoordiantesOptimizer:
    def __init__(self, experiment):
        self.model = experiment.models[0]
        self.vars = self.model.hvar_mgr.all_trainable_weights()
        self.loss = self.model.loss()

        self.spiders = [] #spiders are our tiny solvers.

        self.threads = []

        for _var in self.vars:
            self.spiders.append(VarOptimizer(loss=self.loss, model=self.model, _vars=[_var]))

        # self.spiders.append(VarOptimizer(loss=self.loss, model=self.model, _vars=self.vars))

    def start_threads(self, sess, n_threads):
        for i in range(n_threads):
            t = threading.Thread(target=self.run_thread, args=(sess, ))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)

    def run_thread(self, sess):
        while True:
            for idx in reversed(range(len(self.spiders))):
                #idx = np.random.random_integers(0, len(self.spiders) - 1)
                additional_feed_dict = {}

                # vars_vals = [sess.run(_var) for _var in self.vars]
                #
                #
                # # for _var, val, i in zip(self.vars, vars_vals, range(len(vars_vals))):
                # #     if i != idx:
                # #         additional_feed_dict[_var] = val

                self.spiders[idx].run_iteration(sess, additional_feed_dict=additional_feed_dict)
