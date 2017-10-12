

from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('old_sesop_system_experiments')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il', db_name='old_sesop_system_experiments_db'))


@ex.config
def my_config():
    old_id = -1
    tensorboard_dir = '/home/shai/ExperimentsManager/TensorBoard/' + str(old_id)
    old_results_path = '/home/shai/ExperimentsManager/' + str(old_id)

@ex.automain
def my_main(sesop_batch_mult, sesop_batch_size, b, hidden_layers_sizes, hSize, sesop_freq,
            dataset_size, NORMALIZE_DIRECTIONS, learning_rate_per_node, epochs, base_optimizer, tensorboard_dir, old_results_path,
            subspace_optimizer, fixed_dropout_during_sesop, lr, num_residual_units, model, DISABLE_VECTOR_BREAKING, fixed_bn_during_sesop, nodes, weight_decay_rate, old_id):
    import experiment
    import experiment_results
    import experiments_manager

    mgr = experiments_manager.ExperimentsManager()
    e = mgr.get_experiment_by_id(old_id)
    e = mgr.load_experiment(e)
    results = e.results[0]

    print results


