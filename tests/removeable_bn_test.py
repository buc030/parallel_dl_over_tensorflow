
import tensorflow as tf
from batch_provider import CifarBatchProvider
from model import CifarModel
import experiment
import tf_utils

# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


e = experiment.Experiment(
    {
        'model': 'cifar10',
        'b': 128,
        'lr': 0.01,
        'sesop_batch_size': 0,
        # SV DEBUG
        'sesop_batch_mult': 10,
        'sesop_freq': (1.0 / 391.0),  # sesop every 1 epochs (no sesop)
        'hSize': 1,
        'epochs': 100,
        # saw 5000*100 samples. But if there is a bug, then it is doing only 100 images per epoch
        'nodes': 1,
        # SV DEBUG
        'num_residual_units': 0,
        'optimizer': 'sesop',
        'DISABLE_VECTOR_BREAKING': True,
        'NORMALIZE_DIRECTIONS': True,
        'learning_rate_per_node': False,
        # SV DEBUG
        # 'subspace_optimizer' : 'trust-ncg',
        # 'subspace_optimizer' : 'BFGS',
        'subspace_optimizer': 'natural_gradient',
        'fixed_bn_during_sesop': True,
        'weight_decay_rate': 1.0

    })

batch_prov = CifarBatchProvider(initial_batch_size=100, path='../')
with tf.variable_scope('pid_' + str(0) + '_experiment_' + str(0)):
    with tf.variable_scope("experiment_models") as scope:
        model = CifarModel(e, batch_prov, 0)

var_list = model.hvar_mgr.all_trainable_weights()
cg_var_list = model.hvar_mgr.all_trainable_alphas()

grads = tf.gradients(model.loss(), cg_var_list)
grad_norm = tf.global_norm(grads)


move_op = [tf.assign(var, var + 0.001) for var in var_list]

check_op = tf.add_check_numerics_ops()


#config=tf.ConfigProto(log_device_placement=True)
config=tf.ConfigProto()
with tf.Session(config=config) as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #We must compute the gradeint without BN!!
    #TODO: change bn implementation to remove this limitation!
    fd = model.get_shared_feed(sess, [])
    feed_dicts = [model.get_shared_feed(sess, []) for i in range(10)]




    for i in range(2):

        print 'loss b4 pop: ' + str(tf_utils.avarge_on_feed_dicts(sess, [model.loss()], feed_dicts, {}))
        model.model.pop_bns(sess, feed_dicts)
        print 'loss after pop: ' + str(tf_utils.avarge_on_feed_dicts(sess, [model.loss()], feed_dicts, {}))

        sess.run(move_op)

        print 'loss b4 push: ' + str(tf_utils.avarge_on_feed_dicts(sess, [model.loss()], feed_dicts, {}))
        model.model.push_bns(sess, feed_dicts)
        print 'loss after push: ' + str(tf_utils.avarge_on_feed_dicts(sess, [model.loss()], feed_dicts, {}))

        print '--------------'


# model = MnistModel()
# model.run()
