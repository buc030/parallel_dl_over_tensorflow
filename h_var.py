
import tensorflow as tf
from summary_manager import SummaryManager
import numpy as np


class HVar:
    #this contains all alphas in the graph

    #problem is when someone is doing get_variable, he expect to reuse the same variable
    #So we should pass this class a ready variable
    #def __init__(self, initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None):
    #    var = tf.Variable(initial_value, trainable, collections, validate_shape, caching_device, name, variable_def, dtype, expected_shape, import_scope)

    #we will get a variable named a/b/model_1/d/W/snapshot
    #and we want to create a variable named a/b/model_1/d/W/snapshot (and to share it with named a/b/model_2/d/W/snapshot)
    #So for snapshot, we shouldnt have model_# in the scope

    def __init__(self, var, model):

        #print 'var.name = ' + str(var.name)
        #self.name = var.name.split(":")[0]
        self.name = var.name.split(":")[0].split("/")[-1]
        #print 'self.name = ' + str(self.name)
        self.var = var
        self.model = model

        with tf.variable_scope(self.name):
            self.sub_init(var)

    def sub_init(self, var):
        self.hSize = self.model.experiment.getFlagValue('hSize')
        self.nodes = self.model.experiment.getFlagValue('nodes')
        self.node_id = self.model.node_id

        self.var = var

        self.history = []
        self.history_aplha = []

        self.replicas = []
        self.replicas_aplha = []

        self.next_idx = 0
        self.op_cache = {}
        self.o = None


        with tf.name_scope(self.name + '_subspace'):
            # snapshot is taken after each sesop. after a sesop, the snapshot will contain the value after sesop ran.
            # we need this variable to be shared so we will be able to push the "after sesop value" back into the workers.
            self.last_snapshot = tf.get_variable(initializer=var.initialized_value(), name='snapshot')
            #print 'self.last_snapshot = ' + str(self.last_snapshot.name)
            #node 0 also holds 'nodes - 1' variables. 1 for each other node:
            #the last n-1 replicas are "nodes" replicas.
            with tf.variable_scope('replicas'):
                for i in range(self.nodes - 1):
                    #Note we use tf.getVariable, so thes variables are shared between all nodes.
                    self.replicas.append(tf.get_variable(initializer=np.zeros(var.get_shape(), dtype=np.float32), \
                                                        dtype=var.dtype.base_dtype, name='replica_' + str(i)))
                    #print 'replica: ' + str(self.replicas[-1].name)
                #print 'self.last_snapshot = ' + str(self.last_snapshot.name)

            if self.node_id == 0:
                with tf.name_scope('history'):
                    for i in range(self.hSize):
                        self.history.append(tf.Variable(np.zeros(var.get_shape()),\
                            dtype=var.dtype.base_dtype, name='h_' + str(i)))

                with tf.name_scope('history_alpha'):
                    for i in range(self.hSize):
                        self.history_aplha.append(tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_h_' + str(i)))
                        #SummaryManager.get().add_iter_summary(tf.summary.histogram('alphas_h', self.history_aplha[-1]))

                with tf.name_scope('replicas_aplha'):
                    for i in range(self.nodes - 1):
                        self.replicas_aplha.append(tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_n_' + str(i)))
                        #SummaryManager.get().add_iter_summary(tf.summary.histogram('alphas_n', self.replicas_aplha[-1]))

        self.zero_alpha = None
        if self.node_id == 0:
            for i in range(self.hSize):
                self.push_history_op() #make sure all ops are created

        assert(self.next_idx == 0)

        if self.node_id != 0:
            self.pull_from_master = tf.assign(self.var, self.last_snapshot)
            self.push_to_master = tf.assign(self.replicas[self.node_id - 1], self.out())

    def out(self):
        if self.o is not None:
            return self.o

        with tf.name_scope(self.name + '_out'):
            #return an affine combination of the history vectors
            #and a dictonary to add to feed_dict.
            self.o = self.var

            if self.node_id == 0:
                for r, a in zip(self.history, self.history_aplha):
                    self.o += r * a

                for r, a in zip(self.replicas, self.replicas_aplha):
                    self.o += r*a

            return self.o

    # create an op that puts var of this node into its replica
    #Called before sesop!
    def push_to_master_op(self):
        assert (self.node_id != 0)
        return self.push_to_master

    # this must be called after sesop was executed!
    #Called after sesop!
    def pull_from_master_op(self):
        assert (self.node_id != 0)
        return self.pull_from_master

    # returns an op that updates history and snapshot (executed after optimization on alpha)
    #This must be called when alpahs are non zeros!!!
    def push_history_op(self):
        assert (self.node_id == 0)
        if self.hSize == 0:
            if 0 not in self.op_cache:
                self.op_cache[0] = tf.no_op()
            return self.op_cache[0]


        if self.next_idx not in self.op_cache:
            #print 'HVar Cache Miss, creating the op for var ' + str(self.var.name) + ', idx = ' + str(self.next_idx)
            with tf.name_scope(self.name + '_update'):

                #first we update the original variable to the sesop result
                update_var_op = tf.assign(self.var, self.out())
                with tf.control_dependencies([update_var_op]):
                    #now we update the history (self.var contain the sesop result):
                    update_history_op = tf.assign(self.history[self.next_idx], self.var - self.last_snapshot)
                    with tf.control_dependencies([update_history_op]):
                        #now we update the last_snapshot to be the sesop result
                        update_snapshot_op = tf.assign(self.last_snapshot, self.var)
                        with tf.control_dependencies([update_snapshot_op]):
                            #finally we reset all the alphas (infact we can take this out of the dependecy)
                            #as it only affect self.out()
                            reset_alpha_op = self.zero_alpha_op()
                            self.op_cache[self.next_idx] =\
                                tf.group(update_history_op, update_var_op, update_snapshot_op, reset_alpha_op)

        old_idx = self.next_idx
        self.next_idx = (self.next_idx + 1)%self.hSize

        return self.op_cache[old_idx]

    def zero_alpha_op(self):
        if self.zero_alpha is None:
            group_op = tf.no_op()
            for a in self.replicas_aplha + self.history_aplha:
                group_op = tf.group(group_op, tf.assign(a, np.zeros(1)))
            self.zero_alpha = group_op

        return self.zero_alpha