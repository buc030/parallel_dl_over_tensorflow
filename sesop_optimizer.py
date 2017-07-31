
import tensorflow as tf
import numpy as np

from my_external_optimizer import ScipyOptimizerInterface
from natural_gradient import NaturalGradientOptimizer

from tf_utils import avarge_on_feed_dicts

from tf_utils import lazy_property

class SubspaceOptimizer:
    SUMMARY_BEFORE_ITERATION_KEY = 'SubspaceOptimizerDebugBeforeIter'
    SUMMARY_AFTER_ITERATION_KEY = 'SubspaceOptimizerDebugAfterIter'
    SUMMARY_IN_ITERATION_KEY = 'SubspaceOptimizerDebugInIter'

    def create_history(self, var):
        #return tf.stack([tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype) for i in range(self.history_size)], axis=len(var.get_shape()))
        return [tf.Variable(np.zeros(var.get_shape()), dtype=var.dtype.base_dtype) for i in range(self.history_size)]

    def var_out(self, var):
        #assume vector_breaking == True
        #then every variable, should use its own alpha
        #We need to somehow bound how much alpha can affect the weights. So we normalize the alphas

        #normalizer = tf.sigmoid(self.h_alphas[var])
        #just a soft limit to not let any individual weight explode
        # limit = 2.0
        # normalized_alpha = self.h_alphas[var]
        # normalized_alpha = tf.maximum(tf.minimum(normalized_alpha, limit), -limit)

        #TODO: maybe the limit can be removed
        limit = 2.0 #/len(self.alphas_uniqe)
        #normalized_alpha = self.h_alphas[var] / len(self.alphas_uniqe)
        #self.sgd_distance_moved

        #normalized_alpha = (self.h_alphas[var]*self.alpha_weights)/float(self.history_size)
        #distance_sgd_moved = tf.norm(var - self.last_snapshot[var])
        # normalized_alpha = self.h_alphas[var]/distance_sgd_moved
        normalized_alpha = self.h_alphas[var]
        # normalized_alpha = normalized_alpha / (tf.global_norm([a + self.eps for a in self.alphas_uniqe]))
        # normalized_alpha = tf.maximum(tf.minimum(normalized_alpha, limit), -limit)
        #

        out = var + tf.reduce_sum(tf.stack(self.history[var], axis=len(var.get_shape()))*normalized_alpha, axis=len(var.get_shape()))
        #return (out/tf.norm(out))*tf.norm(var)
        return out

    def build_subspace_graph(self, loss, predictions):

        replacement_ts = {var._snapshot : self.var_out(var) for var in self.orig_var_list}
        #replacement_ts = {var._variable: self.var_out(var) for var in self.orig_var_list}

        if predictions is not None:
            return tf.contrib.graph_editor.graph_replace([loss, predictions], replacement_ts)
        else:
            return tf.contrib.graph_editor.graph_replace([loss], replacement_ts)
        #tf.contrib.graph_editor.graph_replace(target_ts, replacement_ts, dst_scope='', src_scope='',
        #                                      reuse_dst_scope=False)

    @lazy_property
    def history_candicate_norm(self):
        return tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])

    ##################### MINIMIZE OPERATIONS ##########################
    ######## this is by the same order they will be executed! ##########
    ####################################################################

    def update_alpha_weights(self, h_index):
        if h_index in self.update_alpha_weights_ops:
            return self.update_alpha_weights_ops[h_index]

        #we are in h_index
        #so h_index is the most recent, it get weighted h_size
        #(h_index - 1) % h_size is second most recent, it get weighted h_size - 1

        ops = []
        score = float(self.history_size)
        for i in range(self.history_size):
            ops.append(tf.assign(self.alpha_weights[(h_index - i) % self.history_size], 1/score))
            score -= 1.0

        self.update_alpha_weights_ops[h_index] = ops
        return ops

    # 1. add current history
    def store_history(self, h_index):
        if h_index in self.store_history_ops:
            return self.store_history_ops[h_index]

        ops = []
        history_norm = 1.0
        #SV DEBUG
        #SV TODO: the natural gradient experiments that worked used history_norm = 1.0!
        #history_norm = tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])
        for var in self.orig_var_list:
            ops.append(tf.assign(self.history[var][h_index], (var - self.last_snapshot[var])/history_norm))

        self.store_history_ops[h_index] = ops
        return ops

    #2. minimize by alpha

    # 3. move results into original variables:
    # Use of var_out is not allowed after this point!
    @lazy_property
    def move_results_to_original(self):
        ops = []
        for var in self.orig_var_list:
            ops.append(tf.assign(var, self.var_out(var)))
        return ops

    # 4. Fix the history after minimization
    def fix_history_after_sesop(self, h_index):
        return self.store_history(h_index)

    # 5. take snapshot to use next time we are being called, and zero the alphas
    @lazy_property
    def take_snapshot(self):
        ops = []
        for var in self.orig_var_list:
            ops.append(tf.assign(self.last_snapshot[var], var))
        return ops

    @lazy_property
    def zero_alphas(self):
        ops = []
        for a in self.alphas_uniqe:
            ops.append(tf.assign(a, np.zeros(a.get_shape())))
        return ops

    def __init__(self, loss, var_list, history_size, **optimizer_kwargs):
        self.eps = 1e-6
        self.history_size = history_size
        self.orig_var_list = var_list
        self.orig_loss = loss
        self.store_history_ops = {}
        self.update_alpha_weights_ops = {}
        self.optimizer_kwargs = optimizer_kwargs
        self.next_free_h_index = 0
        self.total_iter = 0

        vector_breaking = optimizer_kwargs['VECTOR_BREAKING']
        sesop_method = optimizer_kwargs['sesop_method']
        sesop_options = optimizer_kwargs['sesop_options']
        predictions = optimizer_kwargs['predictions']
        self.normalize_function_during_sesop = optimizer_kwargs['normalize_function_during_sesop']

        assert (vector_breaking == True or vector_breaking == False)


        with tf.name_scope('last_snapshot'):
            #TODO: does this is created in wrong graph??
            self.last_snapshot = {var : tf.Variable(var.initialized_value(), dtype=var.dtype.base_dtype) for var in var_list}

        with tf.name_scope('history'):
            self.history = {var : self.create_history(var) for var in var_list}

        with tf.name_scope('alphas'):
            if vector_breaking == True:
                self.h_alphas = {var: tf.Variable(np.zeros(history_size), dtype=var.dtype.base_dtype) for var in var_list}
            else:
                h_alphas = tf.Variable(np.zeros(history_size), dtype=var_list[0].dtype.base_dtype)
                self.h_alphas = {var: h_alphas for var in var_list}
        self.alphas_uniqe = list(set(self.h_alphas.values()))
        self.alpha_weights = tf.Variable(np.zeros(history_size), dtype=var_list[0].dtype.base_dtype, trainable=False)
        self.sgd_distance_moved = tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])

        with tf.name_scope('sesop_computation_graph'):
            self.new_loss, self.new_predictions = self.build_subspace_graph(loss, predictions)



        self.alpha_grad = tf.gradients(self.new_loss, self.alphas_uniqe)
        self.alpha_grad_norm = tf.global_norm(self.alpha_grad)
        self.grad_norm_placeholder = tf.placeholder(dtype=self.orig_var_list[0].dtype.base_dtype)

        # self.vars_norm = tf.global_norm([var for var in self.orig_var_list])
        # self.var_outs_norm = tf.global_norm([self.var_out(var) for var in self.orig_var_list])

        if sesop_method == 'natural_gradient':
            self.optimizer = NaturalGradientOptimizer(self.new_loss/self.grad_norm_placeholder, self.new_predictions, self.alphas_uniqe, sesop_options)
        else:
            self.optimizer = \
                    ScipyOptimizerInterface(loss=self.new_loss/self.grad_norm_placeholder, var_list=self.alphas_uniqe, \
                                equalities=None ,method=sesop_method, options=sesop_options)

        [self.store_history(h) for h in range(self.history_size)]
        [self.update_alpha_weights(h) for h in range(self.history_size)]
        [self.fix_history_after_sesop(h) for h in range(self.history_size)]
        self.move_results_to_original
        self.zero_alphas
        self.take_snapshot

        with tf.name_scope('after_sesop'):
            tf.summary.scalar('after_sesop_minibatch_loss_normalized', self.new_loss/self.grad_norm_placeholder, [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
            tf.summary.scalar('after_sesop_minibatch_loss', self.orig_loss, [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
            #for h in range(history_size):
            #    tf.summary.scalar('new_vector_in_history_norm_' + str(h), tf.global_norm([self.history[var][h] for var in self.orig_var_list]), [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
            self.sesop_grad_computations = tf.placeholder(tf.int32)
            self.sesop_grad_computations_summary = tf.summary.scalar('sesop_grad_computations', self.sesop_grad_computations, [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
            self.alpha_summary_histogram = tf.summary.histogram('alpha', tf.stack(self.alphas_uniqe), ['alpha_summary_histogram'])

            self.distance_seboost_moved = tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])
            self.distance_sesop_moved = tf.global_norm([self.var_out(var) - var for var in self.orig_var_list])

            self.distance_seboost_moved_summary = tf.summary.scalar('distance_seboost_moved', self.distance_seboost_moved, ['distance_seboost_moved'])
            self.distance_sesop_moved_summary = tf.summary.scalar('distance_sesop_moved', self.distance_sesop_moved, ['distance_sesop_moved'])

            #SGD vector: var - self.last_snapshot[var]
            #SESOP vector: self.var_out(var) - var


            self.sgd_sesop_dot_product = tf.add_n([tf.reduce_sum((var - self.last_snapshot[var])*(self.var_out(var) - var)) for var in self.orig_var_list])
            self.sgd_sesop_dot_product_summary = tf.summary.scalar('sgd_sesop_dot_product', self.sgd_sesop_dot_product, ['sgd_sesop_dot_product'])

        with tf.name_scope('alpha_after_sesop'):
            self.alpha_summary = []
            for i in range(history_size):
                self.alpha_summary.extend([tf.summary.scalar('a', a[i], ['a']) for a in self.alphas_uniqe])
            self.alpha_summary = tf.summary.merge(self.alpha_summary)

        with tf.name_scope('during_sesop'):
            tf.summary.scalar('grad_norm_during_sesop', tf.norm(self.optimizer.get_packed_loss_grad()), [SubspaceOptimizer.SUMMARY_IN_ITERATION_KEY])
            tf.summary.scalar('loss_during_sesop', self.new_loss, [SubspaceOptimizer.SUMMARY_IN_ITERATION_KEY])

        self.summaries_after_iter = tf.summary.merge_all(SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY)
        self.summaries_in_iter = tf.summary.merge_all(SubspaceOptimizer.SUMMARY_IN_ITERATION_KEY)

        self.i_after_iter = 0
        self.i_in_iter = 0




    def set_summary_writer(self, writer):
        self.writer = writer

    #return _sgd_distance_moved, _distance_sesop_moved, _distance_seboost_moved
    def minimize(self, session=None, feed_dicts=None, fetches=None, override_loss_grad_func=None, additional_feed_dict=None):
        if additional_feed_dict is None:
            additional_feed_dict = {}

        print 'history_candicate_norm = ' + str(session.run(self.history_candicate_norm))

        _sgd_distance_moved = session.run(self.sgd_distance_moved)
        # 0. add current history
        session.run(self.store_history(self.next_free_h_index) + self.update_alpha_weights(self.next_free_h_index))
        self.total_iter += 1

        # 1. calculate grad norm to normalize function:
        _alpha_grad_norm = 1.0
        if self.normalize_function_during_sesop:
            _alpha_grad_norm = session.run(self.alpha_grad_norm)

        additional_feed_dict.update({self.grad_norm_placeholder: _alpha_grad_norm})

        print '_alpha_grad_norm = ' + str(_alpha_grad_norm)
        if _alpha_grad_norm < self.eps: # or self.total_iter < self.history_size:
            if _alpha_grad_norm < self.eps:
                print 'alpha grad norm is zero. In local minima with respect to alpha!'
            session.run(self.take_snapshot)
            self.next_free_h_index = (self.next_free_h_index + 1) % self.history_size

            #self.i_in_iter += 10
            self.writer.add_summary(session.run(self.sesop_grad_computations_summary, {self.sesop_grad_computations : 0}), self.i_after_iter)
            self.i_after_iter += 1
            return _sgd_distance_moved, 0.0, _sgd_distance_moved, 0.0



        self._sesop_grad_computations = 0
        def debug_step_callback(alpha):
            _loss, _grad = self.optimizer.loss_grad_func(alpha)
            s = session.run(self.summaries_in_iter, {self.optimizer.get_packed_loss_grad() : _grad, self.new_loss : _loss})
            self.writer.add_summary(s, self.i_in_iter)
            self.i_in_iter += 1

        def debug_loss_callback(_loss, _grad):
            s = session.run(self.summaries_in_iter, {self.optimizer.get_packed_loss_grad() : _grad, self.new_loss : _loss})
            self.writer.add_summary(s, self.i_in_iter)
            self.i_in_iter += 1
            self._sesop_grad_computations += 1


        #2 minimize by alpha
        self.optimizer.minimize(session, feed_dicts=feed_dicts, fetches=fetches, step_callback=None,
                                loss_callback=debug_loss_callback, override_loss_grad_func=override_loss_grad_func, additional_feed_dict=additional_feed_dict)

        additional_feed_dict.update({self.sesop_grad_computations: self._sesop_grad_computations})

        self.writer.add_summary(session.run(self.alpha_summary_histogram), self.i_after_iter)
        self.writer.add_summary(session.run(self.alpha_summary), self.i_after_iter)


        #seperate between different sesop runs
        self.i_in_iter += 10
        #
        # for i in range(10):
        #     s = session.run(self.summaries_in_iter, {self.optimizer._packed_loss_grad : np.zeros(self.optimizer._packed_loss_grad.get_shape()), self.new_loss : 0})
        #     self.writer.add_summary(s, self.i_in_iter)
        #     self.i_in_iter += 1

        print 'loss after minimize = ' + str(avarge_on_feed_dicts(session, [self.new_loss], feed_dicts))


        #tf.global_norm([self.var_out(var) - var for var in self.orig_var_list])
        _distance_sesop_moved = session.run(self.distance_sesop_moved)
        #TODO: save perf
        _sgd_sesop_dot_product = session.run(self.sgd_sesop_dot_product)
        #TODO: remove additional_feed_dict
        self.writer.add_summary(session.run(self.distance_sesop_moved_summary, additional_feed_dict), self.i_after_iter)
        self.writer.add_summary(session.run(self.sgd_sesop_dot_product_summary, additional_feed_dict), self.i_after_iter)


        #3. move results into original variables:
        session.run(self.move_results_to_original)

        print 'loss after move_results_to_original = ' + str(avarge_on_feed_dicts(session, [self.orig_loss], feed_dicts))

        #this is how much sesop + sgd moved
        #tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])
        _distance_seboost_moved = session.run(self.distance_seboost_moved)
        self.writer.add_summary(session.run(self.distance_seboost_moved_summary, additional_feed_dict), self.i_after_iter)


        #4. Fix the history:
        session.run(self.fix_history_after_sesop(self.next_free_h_index))

        print 'loss after fix_history_after_sesop = ' + str(avarge_on_feed_dicts(session, [self.orig_loss], feed_dicts))

        #5. take snapshot to use next time we are being called, and zero the alphas
        session.run(self.take_snapshot)
        print 'loss after take_snapshot = ' + str(avarge_on_feed_dicts(session, [self.orig_loss], feed_dicts))

        session.run(self.zero_alphas)
        print 'loss after zero_alphas = ' + str(avarge_on_feed_dicts(session, [self.orig_loss], feed_dicts))

        additional_feed_dict.update({self.orig_loss : avarge_on_feed_dicts(session, [self.orig_loss], feed_dicts)[0]})
        additional_feed_dict.update({self.new_loss: avarge_on_feed_dicts(session, [self.new_loss], feed_dicts)[0]})

        self.writer.add_summary(session.run(self.summaries_after_iter, additional_feed_dict), self.i_after_iter)
        self.i_after_iter += 1

        self.next_free_h_index = (self.next_free_h_index + 1) % self.history_size

        print '_distance_seboost_moved = ' + str(_distance_seboost_moved)
        print '_distance_sesop_moved = ' + str(_distance_sesop_moved)
        print '_sgd_distance_moved = ' + str(_sgd_distance_moved)


        return _sgd_distance_moved, _distance_sesop_moved, _distance_seboost_moved, _sgd_sesop_dot_product

