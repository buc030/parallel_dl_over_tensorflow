


from seboost import *
import dataset_manager
import experiment
import experiments_manager
import experiment_results
import sys

debug_list = []
def fc_layer(input, n_in, n_out, log, hSize):
    with tf.name_scope('FC'):
        if log:
            W = HVar(tf.random_normal([n_in, n_out]), name='W')
            b = HVar(tf.zeros([n_out]), name='b')
            a = tf.matmul(input, W.out()) + b.out()

            debug_list.append(W)
        else:
            W = tf.Variable(tf.random_normal([n_in, n_out]), name='W')
            b = tf.Variable(tf.zeros([n_out]), name='b')
            a = tf.matmul(input, W) + b

        out = tf.nn.tanh(a)
        
        if log:
            SummaryManager.get().add_iter_summary(tf.summary.histogram('activations_before_tanh', a))
            SummaryManager.get().add_iter_summary(tf.summary.histogram('activations_after_tanh', out))
        
        return out


def build_model(x, y, dim, log=False, hSize=0):
    layers = [fc_layer(x, dim, dim, log, hSize)]
    for i in range(1):
        layers.append(fc_layer(layers[-1], dim, dim, log, hSize))
    layers.append(fc_layer(layers[-1], dim, 1, log, hSize))

    model_out = layers[-1]


    
    #when log is true we build a model for training!
    if log:
        loss_per_sample = tf.squared_difference(model_out, y, name='loss_per_sample')
        loss = tf.reduce_mean(loss_per_sample, name='loss')
        SummaryManager.get().add_iter_summary(tf.summary.scalar('loss', loss))

        return model_out, loss
    #tf.summary.scalar('loss', loss)
    
    return model_out #, loss, train_step



#bs is batch size
#sesop_freq is in (0,1) and is the fraction of sesop iterations.
#i.e., if sesop_freq = 0.1 then we do 1 sesop iteration for each one sgd iteration
#epochs is the number of epochs
def _run_experiment(experiment, file_writer_suffix):
    del debug_list[0:len(debug_list)]

    #SV TODO: remove this hack!
    tf.reset_default_graph()
    SummaryManager.get().reset()
    HVar.reset()
    tf.set_random_seed(1)

    bs = experiment.getFlagValue('b')
    sesop_freq = experiment.getFlagValue('sesop_freq')
    hSize = experiment.getFlagValue('hSize')
    epochs = experiment.getFlagValue('epochs')
    dim = experiment.getFlagValue('dim')
    dataset_size = experiment.getFlagValue('dataset_size')
    training_data, testing_data, training_labels, testing_labels = dataset_manager.DatasetManager().get_random_data(dim, dataset_size)

    print 'Running experiment with bs = ' + str(bs) + ', sesop_freq = ' + str(sesop_freq) + ', hSize = ' + str(hSize)

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    #with tf.Session('grpc://' + tf_server, config=config) as sess:
    with tf.Session(config=config) as sess:
        with tf.name_scope('data'):
            is_training = tf.placeholder(tf.bool,name='is_training') #must be feed with dict_feed.
            is_full_batch = tf.placeholder(tf.bool, name='is_full_batch')

            full_train_input, full_train_labels = tf.cast(tf.constant(training_data, name='train_dataset_x'),\
                tf.float32), tf.cast(tf.constant(training_labels, name='train_dataset_y'), tf.float32)

            full_test_input, full_test_labels = tf.cast(tf.constant(testing_data, name='test_dataset_x'), tf.float32), \
                tf.cast(tf.constant(testing_labels, name='test_dataset_y'), tf.float32)

            def create_training_dataset(bs):
                input, label = tf.train.slice_input_producer([full_train_input, full_train_labels], name='train_slicer')
                batched_input, batched_labels = \
                    tf.train.batch([input, label], batch_size=bs, name='train_batcher')
                return batched_input, batched_labels


            def create_testing_dataset(bs):
                input, label = tf.train.slice_input_producer([full_test_input, full_test_labels], name='test_slicer')
                batched_input, batched_labels = \
                    tf.train.batch([input, label], batch_size=bs, name='test_batcher')
                return batched_input, batched_labels

            #It is very important to call create_training_dataset and create_testing_dataset 
            #create all queues (for train and test)
            train_batched_input, train_batched_labels = create_training_dataset(bs)
            test_batched_input, test_batched_labels = create_testing_dataset(bs)

            batched_input, batched_labels = tf.case([\
                (tf.logical_and(is_training, is_full_batch), lambda: [full_train_input, full_train_labels]),
                (tf.logical_and(is_training, tf.logical_not(is_full_batch)), lambda: [train_batched_input, train_batched_labels]),
                (tf.logical_and(tf.logical_not(is_training), is_full_batch), lambda: [full_test_input, full_test_labels])],
                default=lambda: [test_batched_input, test_batched_labels])

            #batched_input, batched_labels = tf.cond(is_training, lambda: [train_batched_input, train_batched_labels],\
            #    lambda: [test_batched_input, test_batched_labels])


        model_out, loss = build_model(batched_input, batched_labels, dim, True, hSize)

        optimizer = SeboostOptimizer(loss, batched_input, batched_labels)

        #hold acc loss
        with tf.name_scope('loss_accamulator'):
            acc_loss = tf.Variable(0, name='acc_loss', dtype=tf.float32)
            train_loss_summary = tf.summary.scalar('train_loss', acc_loss)
            test_loss_summary = tf.summary.scalar('test_loss', acc_loss)

        iter_summaries = SummaryManager.get().merge_iters()
        optimizer.iter_summaries = iter_summaries


        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #merged_summery = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/generated_data/' + file_writer_suffix)
        writer.add_graph(sess.graph)
        optimizer.writer = writer

        #we must start queue_runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        """
        print 'batched_input, batched_labels = ' + str(np.mean(sess.run([batched_input, batched_labels], feed_dict={is_training: True, is_full_batch: True})[0]))

        print 'Start total_loss = ' + str(sess.run(loss, feed_dict={is_training: True, is_full_batch: True}))
        for i in range(len(debug_list)):
            print 'W[' + str(i) + '] = ' + str(debug_list[i].out().eval())
        """
        #Progress bar:
        for epoch in range(epochs):
            #run 20 steps (full batch optimization to start with)
            print 'epoch #' + str(epoch)
            print 'Computing train error'
            total_loss = sess.run(loss, feed_dict={is_training: True, is_full_batch: True})
            experiment.add_train_error(total_loss)
            #put the accamulated loss into acc_loss node
            writer.add_summary(sess.run(train_loss_summary, feed_dict={acc_loss: total_loss}), epoch)

            print 'Computing test error'
            total_loss = sess.run(loss, feed_dict={is_training: False, is_full_batch: True})
            experiment.add_test_error(total_loss)
            #put the accamulated loss into acc_loss node
            writer.add_summary(sess.run(test_loss_summary, feed_dict={acc_loss: total_loss}), epoch)


            print 'Training'
            optimizer.run_epoch(sess=sess, _feed_dict={is_training: True, is_full_batch: False})


            #writer.flush()
        coord.request_stop()
        coord.join(threads)



def run_experiment(experiment, file_writer_suffix, force_rerun = False):
    if force_rerun == False and experiments_manager.ExperimentsManager.get().load_experiment(experiment) is not None:
        print 'Experiment already ran!'
        return experiments_manager.ExperimentsManager.get().load_experiment(experiment)

    experiments_manager.ExperimentsManager.get().set_current_experiment(experiment)
    _run_experiment(experiment, file_writer_suffix)

    #make experiment presistant
    experiments_manager.ExperimentsManager.get().dump_experiment(experiment)


experiments = {}
i = 0

for sesop_freq in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]:
    for h in [0, 1, 2, 4, 8, 16, 32]:
        for lr in [float(1)/2**i for i in range(10)]:
            experiments[i] = experiment.Experiment(
                {'b': 100,
                 'sesop_freq': sesop_freq,
                 'hSize': h,
                 'epochs': 100, #saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
                 'dim': 10,
                 'lr': lr,
                 'dataset_size': 5000
                 })
            i += 1

for e in experiments.values():
    run_experiment(e, file_writer_suffix='1', force_rerun=False)

comperator = experiment_results.ExperimentComperator(experiments)

#save the experiment results:


import matplotlib.pyplot as plt

comperator.compare(group_by='b')

"""
for e in experiments.values():
    e.results.plotTrainError()
    e.results.plotTestError()
    print e.results.testError
"""
plt.show()




########################################################
