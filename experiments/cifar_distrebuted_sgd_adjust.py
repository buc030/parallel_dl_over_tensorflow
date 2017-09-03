# Imports
import numpy as np
import tensorflow as tf
import collections
import shutil
import tf_utils


from lr_auto_adjust_sgd import SgdAdjustOptimizer
from batch_provider import CifarBatchProvider

from tf_utils import avarge_on_feed_dicts, avarge_n_calls
from resnet_model_original import ResNet, HParams



from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment('cifar_sgd_adjust')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(db_name='cifar_ten_distrebuted'))


@ex.config
def my_config():

    #model:
    num_residual_units = 4
    use_bottleneck = False
    weight_decay_rate = 0.0002
    relu_leakiness = 0.1
    state_of_the_art = False

    lr = 0.1
    batch_size = 100
    n_epochs = 100

    iters_per_adjust = 2
    #iters_per_adjust = 55000
    #iters_per_adjust = (50000/batch_size)/2
    #iters_per_adjust = 50
    #iters_per_adjust = 1
    per_variable = False
    base_optimizer = 'SGD'
    iters_to_wait_before_first_collect = 0

    #for adam
    beta1 = 0.9
    beta2 = 0.999

    #for Adadelta
    rho = 0.95

    tensorboard_dir = tf_utils.allocate_tensorboard_dir()

    #Distrebuted
    num_workers = 2
    num_replicas_to_aggregate = num_workers
    ps_num = 1

    job_name = 'ps'
    task_id = 0

    begin_port = 2222

    ps_hosts = ['localhost:' + str(begin_port + i) for i in range(ps_num)]
    worker_hosts = ['localhost:' + str(begin_port + ps_num + i) for i in range(num_workers)]

@ex.automain
@LogFileWriter(ex)
def my_main(lr, batch_size, n_epochs, iters_per_adjust, per_variable, base_optimizer, beta1, beta2,
            rho, num_residual_units, use_bottleneck, weight_decay_rate, relu_leakiness, state_of_the_art, iters_to_wait_before_first_collect,
            job_name,
            task_id,
            num_replicas_to_aggregate,
            worker_hosts,
            ps_hosts,
            tensorboard_dir):
    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})


    server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
        job_name=job_name,
        task_index=task_id,
        protocol='grpc')

    if job_name == 'ps':
        server.join()
        return

    with tf.device('/job:worker/task:%d' % task_id):

        bp = CifarBatchProvider(batch_size, '/home/shai/tensorflow/parallel_sesop/')
        x, y = bp.batch()

        hps = HParams(batch_size=batch_size,
                      num_classes=10,
                      min_lrn_rate=None,
                      lrn_rate=None,
                      num_residual_units=num_residual_units,
                      use_bottleneck=use_bottleneck,
                      weight_decay_rate=weight_decay_rate,
                      relu_leakiness=relu_leakiness,
                      state_of_the_art=state_of_the_art,
                      optimizer=None)

        model = ResNet(hps, x, y, 'train')
        model._build_model()

        train_accuracy_summary = tf.summary.scalar('train_accuracy', model.accuracy, ['mnist_summary'])
        test_accuracy_summary = tf.summary.scalar('test_accuracy', model.accuracy, ['mnist_summary'])
        train_loss_summary = tf.summary.scalar('train_loss', model.cost, ['mnist_summary'])
        test_loss_summary = tf.summary.scalar('test_loss', model.cost, ['mnist_summary'])

        test_summary = tf.summary.merge([test_loss_summary, test_accuracy_summary])
        train_summary = tf.summary.merge([train_loss_summary, train_accuracy_summary])


        optimizer = SgdAdjustOptimizer(model.cost, bp, tf.trainable_variables(),
                                     lr=lr,
                                     batch_size=batch_size,
                                     train_dataset_size=50000,
                                     iters_per_adjust=iters_per_adjust,
                                     per_variable=per_variable,
                                     base_optimizer=base_optimizer,
                                     beta1=beta1,
                                     beta2=beta2,
                                     rho=rho,
                                     num_replicas_to_aggregate=num_replicas_to_aggregate,
                                     num_workers=len(worker_hosts),
                                     is_chief = task_id==0,
                                     iters_to_wait_before_first_collect=iters_to_wait_before_first_collect)

    chief_queue_runner = optimizer.sgd_optim.get_chief_queue_runner()
    init_tokens_op = optimizer.sgd_optim.get_init_tokens_op()
    if task_id == 0:
        local_init_op = optimizer.sgd_optim.chief_init_op
    else:
        local_init_op = optimizer.sgd_optim.local_step_init_op
    ready_for_local_init_op = optimizer.sgd_optim.ready_for_local_init_op

    with tf.Session(server.target) as sess:

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
        writer.flush()

        optimizer.set_summary_writer(writer)


        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #the chief que runner is not added to QUEUE_RUNNERS
        # Thus we need to start it seperatly
        if task_id == 0:
            chief_queue_runner.create_threads(sess, coord=coord, daemon=True, start=True)
            sess.run(init_tokens_op)

        sess.run([local_init_op, ready_for_local_init_op])

        for epoch in range(n_epochs):
            print 'epoch = ' + str(epoch)
            print 'Write graph into tensorboard into: ' + str(tensorboard_dir)
            ex.info['epoch'] = epoch

            print 'Calculating accuracy...'
            # bp.set_deque_batch_size(sess, 50000/100)
            # writer.add_summary(sess.run(train_accuracy_summary, {model.accuracy: avarge_n_calls(sess, model.accuracy, 100)}), epoch)
            # writer.add_summary(sess.run(train_loss_summary, {model.cost: avarge_n_calls(sess, model.cost, 100)}), epoch)
            #
            # bp.set_data_source(sess, 'test')
            # bp.set_deque_batch_size(sess, 10000/100)
            # writer.add_summary(sess.run(test_accuracy_summary, {model.accuracy: avarge_n_calls(sess, model.accuracy, 100)}), epoch)
            # writer.add_summary(sess.run(test_loss_summary, {model.cost: avarge_n_calls(sess, model.cost, 100)}), epoch)
            # bp.set_data_source(sess, 'train')
            # bp.set_deque_batch_size(sess, batch_size)

            #optimize
            for i in range(50000/batch_size):
                optimizer.run_iter(sess)
            print '---------------'