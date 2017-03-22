


from seboost import *
import sys



def fc_layer(input, n_in, n_out, log, hSize):
    with tf.name_scope('FC'):
        if log:
            W = HVar(tf.Variable(tf.random_normal([n_in, n_out]), name='W'), hSize)
            b = HVar(tf.Variable(tf.zeros([n_out]), name='b'), hSize)
            a = tf.matmul(input, W.out()) + b.out()
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

def generate_random_data(dim, n=5000):
    cov = np.random.rand(dim, dim)
    cov = np.dot(cov, cov.transpose())

    training_data = np.random.multivariate_normal(np.zeros(dim), cov, n)
    testing_data = np.random.multivariate_normal(np.zeros(dim), cov, n)
    
    with tf.name_scope('generating_data'):
        x = tf.placeholder(tf.float32, shape=[None, dim], name='x')
        model_out = build_model(x, None, dim, False)

        #with tf.Session('grpc://' + tf_server, config=config) as sess:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            training_labels = sess.run(model_out, feed_dict={x: training_data})
            testing_labels = sess.run(model_out, feed_dict={x: testing_data})

        return training_data, testing_data, training_labels, testing_labels



#bs is batch size
#sesop_freq is in (0,1) and is the fraction of sesop iterations.
#i.e., if sesop_freq = 0.1 then we do 1 sesop iteration for each one sgd iteration
#epochs is the number of epochs
def run_experiment(bs, sesop_freq, hSize, epochs, file_writer_suffix):
    print 'Running experiment with bs = ' + str(bs) + ', sesop_freq = ' + str(sesop_freq) + ', hSize = ' + str(hSize)
    #return
    dim = 20


    training_data, testing_data, training_labels, testing_labels = generate_random_data(dim, 5000)
    #print training_data.shape
    #print testing_data.shape
    sys.stdout.flush()
    #batch_size
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    #with tf.Session('grpc://' + tf_server, config=config) as sess:
    with tf.Session(config=config) as sess:
        with tf.name_scope('data'):
            is_training = tf.placeholder(tf.bool,name='is_training') #must be feed with dict_feed.

            def create_training_dataset():
                inputs = tf.cast(tf.constant(training_data, name='train_dataset_x'), tf.float32)
                labels = tf.cast(tf.constant(training_labels, name='train_dataset_y'), tf.float32)
                input, label = tf.train.slice_input_producer([inputs, labels], name='train_slicer')
                batched_input, batched_labels = \
                    tf.train.batch([input, label], batch_size=bs, name='train_batcher')
                return batched_input, batched_labels


            def create_testing_dataset():
                inputs = tf.cast(tf.constant(testing_data, name='test_dataset_x'), tf.float32)
                labels = tf.cast(tf.constant(testing_labels, name='test_dataset_y'), tf.float32)

                input, label = tf.train.slice_input_producer([inputs, labels], name='test_slicer')
                batched_input, batched_labels = \
                    tf.train.batch([input, label], batch_size=bs, name='test_batcher')
                return batched_input, batched_labels


            #It is very important to call create_training_dataset and create_testing_dataset 
            #create all queues (for train and test)
            train_batched_input, train_batched_labels = create_training_dataset()
            test_batched_input, test_batched_labels = create_testing_dataset()


            batched_input, batched_labels = tf.cond(is_training, lambda: [train_batched_input, train_batched_labels],\
                lambda: [test_batched_input, test_batched_labels])


        sys.stdout.flush()
        model_out, loss = build_model(batched_input, batched_labels, dim, True, hSize)
        sys.stdout.flush()

        iters_per_epoch = 5000/bs
        sgd_steps = int(1/sesop_freq)
        optimizer = SeboostOptimizer(loss, batched_input, batched_labels, sgd_steps)

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
        sys.stdout.flush()

        for epoch in range(epochs):
            #run 20 steps (full batch optimization to start with)
            print 'epoch #' + str(epoch)
            print 'Computing train error'
            sys.stdout.flush()
            total_loss = 0
            for i in range(iters_per_epoch):
                iter_loss = sess.run(loss, feed_dict={is_training: True})
                total_loss += iter_loss
            #put the accamulated loss into acc_loss node
            writer.add_summary(sess.run(train_loss_summary, feed_dict={acc_loss: total_loss/(5000/bs)}), epoch)

            
            
            print 'Computing test error'
            sys.stdout.flush()
            total_loss = 0
            for i in range(iters_per_epoch):
                iter_loss = sess.run(loss, feed_dict={is_training: False})
                total_loss += iter_loss
            #put the accamulated loss into acc_loss node
            writer.add_summary(sess.run(test_loss_summary, feed_dict={acc_loss: total_loss/(5000/bs)}), epoch)

            
            
            print 'Training'
            sys.stdout.flush()
            total_loss = 0
            for i in range(iters_per_epoch):
                #take a batch:
                batched_input_actual, batched_labels_actual = \
                    sess.run([batched_input, batched_labels], feed_dict={is_training: True})


                #this runs 1 iteration and keeps track of when should it do sesop.
                iter_loss = optimizer.run_sesop_iteration(sess=sess, _feed_dict={is_training: True} ,\
                    sesop_feed_dict=\
                    {is_training: True, batched_input: batched_input_actual, batched_labels: batched_labels_actual})


            #writer.flush()
        coord.request_stop()
        coord.join(threads)




"""
import argparse
import numpy as np
import sys


summaryMgr = SummaryManager()

parser = argparse.ArgumentParser(description='Run an experiment.')
parser.add_argument('--batch_size', type=int, nargs=1, required=True, help='batch size')
parser.add_argument('--sesop_freq', type=float, nargs=1, required=True, help='Sesop frequancy')
parser.add_argument('--h_size', type=int, nargs=1, required=True, help='Sesop history size')
parser.add_argument('--epochs', type=int, nargs=1, required=True, help='Number of epochs to run')
parser.add_argument('--tf_server', nargs=1, required=True, help='Something like localhost:2222')
parser.add_argument('--log_dir_suffix', type=int, nargs=1, required=True, help='The index of the logdir this will be saved in')

args = parser.parse_args()
tf_server = args.tf_server[0]



import pickle
num = args.log_dir_suffix[0]

print args


import tensorflow as tf
#contrib/opt/python/training/external_optimizer.py
print tf.contrib.opt.ScipyOptimizerInterface



#put the info of this run into descr file so we will be able to do lookup by parameters later!
print 'using num = ' + str(num)
with open('/tmp/generated_data/' + str(num) + '.descr', 'wb') as f:	
	pickle.dump(['started', args], f)

sys.stdout.flush()
run_experiment(bs=args.batch_size[0], sesop_freq=args.sesop_freq[0],\
	 hSize=args.h_size[0], epochs=args.epochs[0], file_writer_suffix=str(num))

with open('/tmp/generated_data/' + str(num) + '.descr', 'wb') as f:
	pickle.dump(['done', args], f)

"""


########################################################
