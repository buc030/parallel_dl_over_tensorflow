
import tensorflow as tf
import re
import numpy as np

class SharedVariablesManager:
    def __init__(self):
        pass

    snapshots = {}
    replicas = {}

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