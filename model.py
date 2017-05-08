
import tensorflow as tf
from summary_manager import SummaryManager
from h_var import HVar
import utils
from utils import check_create_dir
import numpy as np
import debug_utils

class FCLayer:
    def __init__(self, input, n_in, n_out, model, prefix, activation=True):
        with tf.variable_scope(prefix):

            low = -np.sqrt(6.0 / (n_in + n_out))  # use 4 for sigmoid, 1 for tanh activation
            high = np.sqrt(6.0 / (n_in + n_out))

            #print 'prefix = ' + str(prefix)
            self.W = model.hvar_mgr.create_var(tf.Variable(tf.random_uniform([n_in, n_out], minval=low, maxval=high, dtype=tf.float32), name='W'))
            self.b = model.hvar_mgr.create_var(tf.Variable(tf.zeros([n_out]), name='b'))
            a = tf.matmul(input, self.W.out()) + self.b.out()
            if activation == False:
                self.out = a
            else:
                self.out = tf.nn.tanh(a)

    def print_params(self, sess):
        print 'W = ' + str(sess.run(self.W.var))
        #print 'b = ' + str(sess.run(self.b.var))
        print 'W.out = ' + str(sess.run(self.W.out()))



#This holds the model symbols
class Model(object):

    #in default case, accuracy is just the loss
    def accuracy(self):
        return self._loss

    def calc_train_accuracy(self, sess, batch_size, train_dataset_size):
        train_error = np.zeros(1)
        self.batch_provider.set_data_source(sess, 'train')

        for i in range((train_dataset_size / 1) / batch_size):
            train_error += np.array(sess.run(self.accuracy()))
        train_error /= float((train_dataset_size / 1) / batch_size)
        return train_error

    def dump_debug(self, sess, suffix):
        with open('debug_' + suffix, 'w') as f:

            for debug_hvar in self.hvar_mgr.all_hvars:
                f.write('debug_hvar.out() = ' + str(sess.run(debug_hvar.out())) + '\n')
                f.write('---------------------')
            f.flush()

    def print_layer_params(self, sess, i):
         self.layers[i].print_params(sess)

    class HVarManager:
        def __init__(self, model):
            self.all_hvars = []
            self.model = model
            tf.set_random_seed(895623)

        def create_var(self, var):
            res = HVar(var, self.model)
            self.all_hvars.append(res)

            #print 'res.var.name = ' + str(res.var.name)
            name = res.var.name.split(":")[0].split("/")[-1]
            #print 'name = ' + str(name)
            with tf.name_scope(name):
                for alpha in res.history_aplha:
                    tmp = tf.summary.histogram('history_aplhas', alpha)
                    #print 'alpha name = ' + str(tmp.name)
                    self.model.summary_mgr.add_iter_summary(tmp)

            return res

        def reset(self):
            self.all_hvars = []

        def assert_alphas_are_zero(self):
            res = []
            for hvar in self.all_hvars:
                res.append(hvar.assert_alphas_are_zero())
            return res


        # the alphas from sesop (the coefitients that choose the history vector)
        def all_trainable_alphas(self):
            alphas = []
            for hvar in self.all_hvars:
                alphas.extend(hvar.replicas_aplha + hvar.history_aplha)
                #alphas.extend(hvar.history_aplha)

            res = []
            #remoove duplications
            for var in alphas:
                if var not in res:
                    res.append(var)

            return res

        # all the regular weights to be trained
        def all_trainable_weights(self):
            weights = []
            for hvar in self.all_hvars:
                weights.append(hvar.var)
            return weights

        def normalize_directions_ops(self):
            if not hasattr(self, '__normalize_directions_ops'):
                self.__normalize_directions_ops = {}
                for index in self.model.hSize:
                    terms = []
                    for hvar in self.all_hvars:
                        terms.append(hvar.history[index])

                    norm = tf.global_norm(terms)
                    vectors_to_normalize = {} #remove duplicates using a hash table
                    for hvar in self.all_hvars:
                        for d in hvar.history + hvar.replicas:
                            vectors_to_normalize[d] = 0

                    operations = []
                    for d in vectors_to_normalize.keys():
                        #make all direction vectors to be in the same size of the last progress:
                        operations.append(tf.assign(d, (d / tf.norm(d))*norm ))

                    self.__normalize_directions_ops[index] = operations

            prev_index = None
            for hvar in self.all_hvars:
                index = hvar.get_index_of_last_direction()
                assert (prev_index is None or prev_index == index)
                prev_index = index

            assert (prev_index is not None)
            return self.__normalize_directions_ops[prev_index]
        def all_history_update_ops(self):
            b4_sesop = []
            after_sesop = []
            for hvar in self.all_hvars:
                b4, after = hvar.update_history_op()
                b4_sesop.append(b4)
                after_sesop.append(after)

            return b4_sesop, after_sesop

        def all_zero_alpha_ops(self):
            res = []
            for hvar in self.all_hvars:
                res.append(hvar.zero_alpha_op())

            return res

    def dump_checkpoint(self, sess):
        path = self.experiment.get_model_checkpoint_dir(self.node_id)
        if self.saver is None:

            check_create_dir(path)
            self.saver = tf.train.Saver(self.hvar_mgr.all_trainable_alphas() + self.hvar_mgr.all_trainable_weights())

        self.saver.save(sess, path + '/model.save', global_step=None)

    def init_from_checkpoint(self, sess):
        path = self.experiment.get_model_checkpoint_dir(self.node_id)
        if self.saver is None:
            self.saver = tf.train.Saver(self.hvar_mgr.all_trainable_alphas() + self.hvar_mgr.all_trainable_weights())


        self.saver.restore(sess, path)


    #self dontate a batch to be used by all.
    def get_shared_feed(self, sess, models):
        x, y = sess.run([self.input, self.label])
        res = {self.input: x, self.label: y}
        for m in models:
            res[m.input] = x
            res[m.label] = y

        return res


    def __init__(self, experiment, batch_provider, node_id):
        self.experiment = experiment
        self.node_id = node_id
        self.nodes = experiment.getFlagValue('nodes')
        self.hSize = experiment.getFlagValue('hSize')
        self.saver = None
        #print 'node_id = ' + str(node_id)
        #print '-------------------------'

        self.hvar_mgr = Model.HVarManager(self)  # every experiment has its own hvar collection and summary collection.

        self.tensorboard_dir = self.experiment.get_model_tensorboard_dir(self.node_id)
        self.summary_mgr = SummaryManager(self.tensorboard_dir)

        self.batch_provider = batch_provider

        self.i = 0
        self.j = 0

        utils.printInfo(' self.batch_provider.batch() = ' + str( self.batch_provider.batch()))

        self.input, self.label = self.batch_provider.batch()


        # if not (self.experiment.getFlagValue('hSize') == 0 ) and self.node_id == 0:
        #     #print 'experiment = ' + str(experiment)
        #     self.mergered_summeries = self.summary_mgr.merge_iters()

        utils.printInfo('Dumping into tensorboard ' + str(self.tensorboard_dir))


    def push_to_master_op(self):
        assert (self.node_id != 0)
        return [hvar.push_to_master_op() for hvar in self.hvar_mgr.all_hvars]

    def pull_from_master_op(self):
        assert (self.node_id != 0)
        return [hvar.pull_from_master_op() for hvar in self.hvar_mgr.all_hvars]

    def loss(self):
        return self._loss

    def get_batch_provider(self):
        return self.batch_provider

    def get_inputs(self):
        return self.input, self.label




#This holds the model symbols
class SimpleModel(Model):

    def __init__(self, experiment, batch_provider, node_id):
        super(SimpleModel, self).__init__(experiment, batch_provider, node_id)
        assert (experiment.getFlagValue('model') == 'simple')

        hidden_layers_sizes = experiment.getFlagValue('hidden_layers_sizes')
        input_dim = experiment.getInputDim()

        #build layers:
        with tf.variable_scope('model_' + str(self.node_id)):
            self.layers = [FCLayer(self.input, input_dim, hidden_layers_sizes[1], self, 'FC_'  + str(0))]

            for i in range(1, len(hidden_layers_sizes) - 1):
                self.layers.append(
                    FCLayer(self.layers[-1].out, hidden_layers_sizes[i],
                            hidden_layers_sizes[i + 1], self, 'FC_' + str(len(self.layers)), i == len(hidden_layers_sizes) - 2))

            self.model_out = self.layers[-1].out

            # when log is true we build a model for training!

            loss_per_sample = tf.squared_difference(self.model_out, self.label, name='loss_per_sample')
            self._loss = tf.reduce_mean(loss_per_sample, name='loss')

            self.build_train_op()

            self.input_norm = tf.global_norm([self.input, self.label])


    def get_extra_train_ops(self):
        return []

    def div_learning_rate(self, sess, factor):
        sess.run(self.div_learning_rate_op, feed_dict={self.lrn_rate_divide_factor: factor})

    def build_train_op(self):
        self.lrn_rate = tf.Variable(initial_value=self.experiment.getFlagValue('lr'), trainable=False, dtype=tf.float32,
                               name='model_start_learning_rate')  # tf.constant(self.hps.lrn_rate, tf.float32)

        self.lrn_rate_divide_factor = tf.placeholder(dtype=tf.float32)
        self.div_learning_rate_op = tf.assign(self.lrn_rate, self.lrn_rate/self.lrn_rate_divide_factor)

        trainable_variables = self.hvar_mgr.all_trainable_weights()
        grads = tf.gradients(self._loss, trainable_variables)
        #grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]

        if self.experiment.getFlagValue('base_optimizer') == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.experiment.getFlagValue('base_optimizer') == 'momentom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.experiment.getFlagValue('base_optimizer') == 'adagrad':
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdagradDAOptimizer(self.lrn_rate, global_step)
        else:
            assert(False)

        self.grad_norm = tf.global_norm(grads)
        self.weights_norm = tf.global_norm(trainable_variables)

        if debug_utils.DEBUG_PRINT_GRADIENT_NORMS:
            grads[0] = tf.Print(grads[0], [tf.global_norm(grads)], 'Norm of gradient of node ' + str(self.node_id) + ' : ')

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables), name='train_step')

        train_ops = [apply_op] + self.get_extra_train_ops()
        self._train_op = tf.group(*train_ops)


    def train_op(self):
        return [self._train_op]

from resnet_model import ResNet, HParams

#This holds the model symbols
class MnistModel(Model):

    def __init__(self, experiment, batch_provider, node_id):
        super(MnistModel, self).__init__(experiment, batch_provider, node_id)
        assert (experiment.getFlagValue('model') == 'mnist')

        hps = HParams(batch_size=None,
                      num_classes=10,
                      min_lrn_rate=0.0001,
                      lrn_rate=0.1,
                      num_residual_units=5,
                      use_bottleneck=False,
                      weight_decay_rate=0.0002,
                      relu_leakiness=0.1,
                      optimizer=None)

        def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 custom_getter=None):

            var = tf.get_variable(name, shape, dtype, initializer, regularizer, trainable, collections, caching_device,
                            partitioner, validate_shape, custom_getter)
            h_var = self.hvar_mgr.create_var(var)
            return h_var.out()

        #filter_size, filter_size, in_filters, out_filters

        ##[?,1,4,4], [3,3,1,16].
        self.input_after_reshape = tf.reshape(self.input, [-1, 4, 4, 1])
        self.model = ResNet(hps, self.input_after_reshape, self.label, 'train', 1, get_variable)
        self.model._build_model()
        self._loss = self.model.cost


#This holds the model symbols
class CifarModel(Model):

    def get_extra_train_ops(self):
        return self.model._extra_train_ops

    def accuracy(self):
        return self.model._accuracy

    def train_op(self):
        return [self.model.train_op]

    def __init__(self, experiment, batch_provider, node_id):
        super(CifarModel, self).__init__(experiment, batch_provider, node_id)
        assert (experiment.getFlagValue('model') == 'cifar10')


        def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 custom_getter=None):

            #+ '_node_' + str(self.node_id)
            var = tf.get_variable(name, shape, dtype, initializer, regularizer, trainable, collections, caching_device,
                            partitioner, validate_shape, custom_getter)
            h_var = self.hvar_mgr.create_var(var)
            return h_var.out()

        def get_h_variable(name,
                         shape=None,
                         dtype=None,
                         initializer=None,
                         regularizer=None,
                         trainable=True,
                         collections=None,
                         caching_device=None,
                         partitioner=None,
                         validate_shape=True,
                         custom_getter=None):

            var = tf.get_variable(name, shape, dtype, initializer, regularizer, trainable, collections, caching_device,
                                  partitioner, validate_shape, custom_getter)
            h_var = self.hvar_mgr.create_var(var)
            return h_var

        #filter_size, filter_size, in_filters, out_filters

        ##[?,1,4,4], [3,3,1,16].
        #self.input_after_reshape = tf.reshape(self.input, [-1, 4, 4, 1])
        hps = HParams(batch_size=None,
                      num_classes=10,
                      min_lrn_rate=0.0001,
                      lrn_rate=self.experiment.getFlagValue('lr'),
                      num_residual_units=self.experiment.getFlagValue('num_residual_units'),
                      use_bottleneck=False,
                      weight_decay_rate=0.0002,
                      relu_leakiness=0.1,
                      optimizer=self.experiment.getFlagValue('optimizer'))

        #self.input = tf.reshape(self.input, [-1, 3, 32, 32])
        with tf.variable_scope('model_' + str(self.node_id)):
            self.model = ResNet(hps, self.input, self.label, 'train', 3, get_variable, get_h_variable, self.hvar_mgr)
            self.model._build_model()

            #self.model._extra_train_ops.append(self.stage)
            self.model._build_train_op()

        self._loss = self.model.cost
        self.model._accuracy = self.model.accuracy #tf.group(*[self.model.accuracy, self.stage])

