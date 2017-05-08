
import tensorflow as tf
import re
import numpy as np
import debug_utils

class SharedVariablesManager:
    def __init__(self):
        pass

    snapshots = {}
    replicas = {}
    history_aplhas = {}
    replicas_aplhas = {}

    @classmethod
    def remove_model_index_from_name(cls, name):
        return re.sub('model_[0-9]+', '', name, count=1)


    #model: the requesting model
    #var: the var to create the shared copy of
    @classmethod
    def get_snapshot(cls, model, var):

        name = SharedVariablesManager.remove_model_index_from_name(var.name)
        if model.node_id == 0:
            SharedVariablesManager.snapshots[name] = tf.get_variable(initializer=var.initialized_value(), name='snapshot')

        return SharedVariablesManager.snapshots[name]

    @classmethod
    def get_replicas(cls, model, var):
        name = SharedVariablesManager.remove_model_index_from_name(var.name)
        if model.node_id == 0:
            nodes = model.experiment.getFlagValue('nodes')
            SharedVariablesManager.replicas[name] = []
            for i in range(nodes - 1):
                # Note we use tf.getVariable, so thes variables are shared between all nodes.
                SharedVariablesManager.replicas[name].append(tf.get_variable(initializer=np.zeros(var.get_shape(), dtype=np.float32), \
                                                     dtype=var.dtype.base_dtype, name='replica_' + str(i)))
        return SharedVariablesManager.replicas[name]

    @classmethod
    def get_history_aplha(cls, model, var):
        #No vector breaking
        if debug_utils.DISABLE_VECTOR_BREAKING == False or model not in SharedVariablesManager.history_aplhas:
            hSize = model.experiment.getFlagValue('hSize')
            history_aplha = []
            for i in range(hSize):
                history_aplha.append(tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_h_' + str(i)))
            SharedVariablesManager.history_aplhas[model] = history_aplha

        return SharedVariablesManager.history_aplhas[model]


    @classmethod
    def get_replicas_aplha(cls, model, var):
        # No vector breaking
        if debug_utils.DISABLE_VECTOR_BREAKING == False or model not in SharedVariablesManager.replicas_aplhas:
            nodes = model.experiment.getFlagValue('nodes')
            replicas_aplha = []
            for i in range(nodes - 1):
                replicas_aplha.append(
                    tf.Variable(np.zeros(1), dtype=var.dtype.base_dtype, name='alpha_n_' + str(i)))

            SharedVariablesManager.replicas_aplhas[model] = replicas_aplha

        return SharedVariablesManager.replicas_aplhas[model]