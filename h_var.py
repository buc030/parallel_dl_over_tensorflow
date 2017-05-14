
import tensorflow as tf
import numpy as np
import debug_utils

from tensorflow.python.ops.control_flow_ops import with_dependencies

from SharedVariablesManager import SharedVariablesManager

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

        print 'var.name = ' + str(var.name)
        print 'self.name = ' + str(self.name)

        with tf.variable_scope(self.name + '_subspace'):
            self.sub_init(var)

    def sub_init(self, var):
        self.hSize = self.model.experiment.getFlagValue('hSize')
        self.nodes = self.model.experiment.getFlagValue('nodes')
        self.node_id = self.model.node_id

        self.var = var
        self.history = []
        self.history_aplha = []

        self.replicas_aplha = []

        self.next_idx = 0
        self.op_cache = {}
        self.o = None
        if self.nodes == 1 and self.hSize == 0:
            return

        # snapshot is taken after each sesop. after a sesop, the snapshot will contain the value after sesop ran.
        # we need this variable to be shared so we will be able to push the "after sesop value" back into the workers.
        self.last_snapshot = SharedVariablesManager.get_snapshot(self.model, var)
            #tf.get_variable(initializer=var.initialized_value(), name='snapshot')
        self.replicas = SharedVariablesManager.get_replicas(self.model, var)


        if self.node_id == 0:
            with tf.name_scope('history'):
                for i in range(self.hSize):
                    self.history.append(tf.Variable(np.zeros(var.get_shape()),\
                            dtype=var.dtype.base_dtype, name='h_' + str(i)))

            with tf.name_scope('history_alpha'):
                self.history_aplha = SharedVariablesManager.get_history_aplha(self.model, var)

                # for i in range(self.hSize):
                #     self.history_aplha.append(tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_h_' + str(i)))
                #         #SummaryManager.get().add_iter_summary(tf.summary.histogram('alphas_h', self.history_aplha[-1]))
                #     tf.summary.histogram('alphas_h', self.history_aplha[-1])

            with tf.name_scope('replicas_aplha'):
                self.replicas_aplha = SharedVariablesManager.get_replicas_aplha(self.model, var)
                # for i in range(self.nodes - 1):
                #     self.replicas_aplha.append(
                #         tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_n_' + str(i)))
                    # SummaryManager.get().add_iter_summary(tf.summary.histogram('alphas_n', self.replicas_aplha[-1]))
                    #tf.summary.histogram('alpha_n_', self.replicas_aplha[-1])

        self.zero_alpha = None
        if self.node_id == 0:
            self.assert_alphas_are_zero()
            self.zero_alpha_op()
            self.update_history_op()
            for i in range(self.hSize):
                self.update_history_op() #make sure all ops are created

        if self.node_id != 0:
            with tf.name_scope('pull_from_master'):
                self.pull_from_master = tf.assign(self.var, self.last_snapshot)
            with tf.name_scope('push_to_master'):
                self.push_to_master = tf.assign(self.replicas[self.node_id - 1], self.out() - self.last_snapshot)



    def out(self):
        if self.o is not None:
            return self.o

        with tf.name_scope(self.name + '_out'):
            #return an affine combination of the history vectors
            if self.hSize == 0 and self.nodes == 1:

                self.o = self.var

                if debug_utils.DEBUG_PRINT_INTERMEDIANTE_TENSORS:
                    self.o = tf.Print(self.o, [self.o], 'var = ' + str(self.var.name) + ', node_id = ' + str(self.node_id))
                return self.o

            if self.node_id == 0:
                terms = [self.var]
                for r, a in zip(self.history, self.history_aplha):
                    terms.append(r * a)

                for r, a in zip(self.replicas, self.replicas_aplha):
                    terms.append(r * a)

                self.o = tf.add_n(terms)

                if debug_utils.DEBUG_PRINT_INTERMEDIANTE_TENSORS:
                    self.o = tf.Print(self.o, [self.o], 'var = ' + str(self.var.name) + ', node_id = ' + str(self.node_id))
                return self.o

            self.o = self.var

            if debug_utils.DEBUG_PRINT_INTERMEDIANTE_TENSORS:
                self.o = tf.Print(self.o, [self.o], 'var = ' + str(self.var.name) + ', node_id = ' + str(self.node_id))

            return self.o

    def assert_alphas_are_zero(self):
        return []
        if hasattr(self, 'assert_alphas_are_zero_op'):
            return self.assert_alphas_are_zero_op

        res = []
        for a in self.replicas_aplha + self.history_aplha:
            res.append(tf.Assert(tf.equal(a[0], 0), [a]))

        self.assert_alphas_are_zero_op = res
        return res

    def get_index_of_last_direction(self):
        return self.next_idx

    #return an op that pushes the current progress into history, we need to do this before we optimize by alpha
    #To approximly maintain the expanding mandifold property.
    # This must be called when alpahs are zeros!!!
    def update_history_before_sesop_op(self):
        assert (self.node_id == 0)

        #return [tf.assign(self.history[self.next_idx], self.out() - self.last_snapshot)]
        #put the avrage progress inside the history (we will also have each of the progresses seperatly in the 'replicas' )
        terms = [(self.out() - self.last_snapshot)]

        for r in self.replicas:
            terms.append(r)

        assign_op = tf.assign(self.history[self.next_idx], tf.add_n(terms)/self.nodes)

        #Now all the alphas are zero, and we want to start optimization from the avarage of the progresses (which is history[self.next_idx] at the moment)
        #so we actually need var = snapshot + history[self.next_idx]
        return [assign_op, with_dependencies(terms + [assign_op], tf.assign(self.var, self.last_snapshot + assign_op))]


    # create an op that puts var of this node into its replica
    # Called before sesop!
    def push_to_master_op(self):
        assert (self.node_id != 0)
        return self.push_to_master

    # this must be called after sesop was executed!
    # Called after sesop!
    def pull_from_master_op(self):
        assert (self.node_id != 0)
        return self.pull_from_master

    #return 2 ops:
    # 1. an op that pushes the current progress into history, we need to do this before we optimize by alpha
    # 2. an op that updates history and snapshot (executed after optimization on alpha)
    #This must be called when alpahs are non zeros!!!
    def update_history_op(self):
        assert (self.node_id == 0)
        if self.hSize == 0:
            if 0 not in self.op_cache:
                if self.nodes > 1:
                    update_var_op = tf.assign(self.var, self.out())
                    update_snapshot_op = tf.assign(self.last_snapshot, update_var_op)
                    self.op_cache[0] = [tf.no_op(), [update_var_op, update_snapshot_op]]
                else:
                    self.op_cache[0] = [tf.no_op(), tf.no_op()]

            #self.next_idx = (self.next_idx + 1) % self.hSize
            return self.op_cache[0]

        if self.next_idx not in self.op_cache:
            self.op_cache[self.next_idx] = [self.update_history_before_sesop_op()]
            #print 'HVar Cache Miss, creating the op for var ' + str(self.var.name) + ', idx = ' + str(self.next_idx)
            with tf.name_scope(self.name + '_update'):

                #first we update the original variable to the sesop result
                update_var_op = tf.assign(self.var, self.out())
                #update_var_op = tf.Print(input_=update_var_op, data=[self.var], message='First stage')

                # now we update the history (self.var contain the sesop result):
                update_history_op = with_dependencies([update_var_op], tf.assign(self.history[self.next_idx], update_var_op - self.last_snapshot))
                #update_history_op = tf.Print(input_=update_history_op, data=[self.var], message='Second stage')

                # now we update the last_snapshot to be the sesop result
                update_snapshot_op = with_dependencies([update_history_op], tf.assign(self.last_snapshot, update_var_op))


                # self.op_cache[self.next_idx].append(
                #     [update_history_op, update_var_op, update_snapshot_op, reset_alpha_op])
                self.op_cache[self.next_idx].append(
                     [update_var_op, update_history_op, update_snapshot_op])

        old_idx = self.next_idx
        self.next_idx = (self.next_idx + 1)%self.hSize

        return self.op_cache[old_idx]

    # finally we reset all the alphas (infact we can take this out of the dependecy)
    # as it only affect self.out()
    def zero_alpha_op(self):
        if self.zero_alpha is None:
            group_op = tf.no_op()
            for a in self.replicas_aplha + self.history_aplha:
                group_op = tf.group(group_op, tf.assign(a, np.zeros(1)))
            self.zero_alpha = group_op

        return self.zero_alpha