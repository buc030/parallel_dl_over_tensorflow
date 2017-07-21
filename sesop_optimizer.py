
import tensorflow as tf
import numpy as np

from my_external_optimizer import ScipyOptimizerInterface

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
        normalized_alpha = self.h_alphas[var]
        # normalized_alpha = normalized_alpha / (tf.global_norm([a + self.eps for a in self.alphas_uniqe]))
        # normalized_alpha = tf.maximum(tf.minimum(normalized_alpha, limit), -limit)
        #
        out = var + tf.reduce_sum(tf.stack(self.history[var], axis=len(var.get_shape()))*normalized_alpha, axis=len(var.get_shape()))
        #return (out/tf.norm(out))*tf.norm(var)
        return out

    def build_subspace_graph(self):

        replacement_ts = {var._snapshot : self.var_out(var) for var in self.orig_var_list}
        #replacement_ts = {var._variable: self.var_out(var) for var in self.orig_var_list}
        return tf.contrib.graph_editor.graph_replace(self.orig_loss, replacement_ts)
        #tf.contrib.graph_editor.graph_replace(target_ts, replacement_ts, dst_scope='', src_scope='',
        #                                      reuse_dst_scope=False)

    @lazy_property
    def history_candicate_norm(self):
        return tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])

    ##################### MINIMIZE OPERATIONS ##########################
    ######## this is by the same order they will be executed! ##########
    ####################################################################

    # 1. add current history
    def store_history(self, h_index):
        if h_index in self.store_history_ops:
            return self.store_history_ops[h_index]

        ops = []
        #history_norm = 1.0
        history_norm = tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])
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
        self.optimizer_kwargs = optimizer_kwargs
        self.next_free_h_index = 0
        self.total_iter = 0

        vector_breaking = optimizer_kwargs['VECTOR_BREAKING']
        sesop_method = optimizer_kwargs['sesop_method']
        sesop_options = optimizer_kwargs['sesop_options']


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

        with tf.name_scope('sesop_computation_graph'):
            self.new_loss = self.build_subspace_graph()



        self.alpha_grad = tf.gradients(self.new_loss, self.alphas_uniqe)
        self.alpha_grad_norm = tf.global_norm(self.alpha_grad)
        self.grad_norm_placeholder = tf.placeholder(dtype=self.orig_var_list[0].dtype.base_dtype)

        # self.vars_norm = tf.global_norm([var for var in self.orig_var_list])
        # self.var_outs_norm = tf.global_norm([self.var_out(var) for var in self.orig_var_list])

        #TODO: move to config
        #SV DEBUG
        sesop_options = {'maxiter': 200, 'gtol': 1e-6}
        sesop_method = 'BFGS'
        self.optimizer = \
                ScipyOptimizerInterface(loss=self.new_loss/self.grad_norm_placeholder, var_list=self.alphas_uniqe, \
                            equalities=None ,method=sesop_method, options=sesop_options)

        [self.store_history(h) for h in range(self.history_size)]
        [self.fix_history_after_sesop(h) for h in range(self.history_size)]
        self.move_results_to_original
        self.zero_alphas
        self.take_snapshot

        tf.summary.scalar('after_sesop_minibatch_loss_normalized', self.new_loss/self.grad_norm_placeholder, [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
        tf.summary.scalar('after_sesop_minibatch_loss', self.orig_loss, [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
        #for h in range(history_size):
        #    tf.summary.scalar('new_vector_in_history_norm_' + str(h), tf.global_norm([self.history[var][h] for var in self.orig_var_list]), [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
        self.sesop_grad_computations = tf.placeholder(tf.int32)
        tf.summary.scalar('sesop_grad_computations', self.sesop_grad_computations, [SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY])
        self.alpha_summary = tf.summary.histogram('alpha', tf.stack(self.alphas_uniqe), ['alpha_summary'])



        tf.summary.scalar('grad_norm_during_sesop', tf.norm(self.optimizer._packed_loss_grad), [SubspaceOptimizer.SUMMARY_IN_ITERATION_KEY])
        tf.summary.scalar('loss_during_sesop', self.new_loss, [SubspaceOptimizer.SUMMARY_IN_ITERATION_KEY])

        self.summaries_after_iter = tf.summary.merge_all(SubspaceOptimizer.SUMMARY_AFTER_ITERATION_KEY)
        self.summaries_in_iter = tf.summary.merge_all(SubspaceOptimizer.SUMMARY_IN_ITERATION_KEY)

        self.i_after_iter = 0
        self.i_in_iter = 0

        self.distance_seboost_moved = tf.global_norm([var - self.last_snapshot[var] for var in self.orig_var_list])
        self.distance_sesop_moved = tf.global_norm([var - self.var_out(var) for var in self.orig_var_list])

        self.distance_seboost_moved_summary = tf.summary.scalar('distance_seboost_moved', self.distance_seboost_moved, ['distance_seboost_moved'])
        self.distance_sesop_moved_summary = tf.summary.scalar('distance_sesop_moved', self.distance_sesop_moved, ['distance_sesop_moved'])


    def set_summary_writer(self, writer):
        self.writer = writer

    #return the amount we should have moved since last call
    def minimize(self, session=None, feed_dicts=None, fetches=None, override_loss_grad_func=None, additional_feed_dict=None):
        if additional_feed_dict is None:
            additional_feed_dict = {}

        print 'history_candicate_norm = ' + str(session.run(self.history_candicate_norm))

        # 0. add current history
        session.run(self.store_history(self.next_free_h_index))
        self.total_iter += 1

        # 1. calculate grad norm to normalize function:
        _alpha_grad_norm = session.run(self.alpha_grad_norm)

        additional_feed_dict.update({self.grad_norm_placeholder: _alpha_grad_norm})

        print '_alpha_grad_norm = ' + str(_alpha_grad_norm)
        if _alpha_grad_norm < self.eps: # or self.total_iter < self.history_size:
            if _alpha_grad_norm < self.eps:
                print 'alpha grad norm is zero. In local minima with respect to alpha!'
            session.run(self.take_snapshot)
            self.next_free_h_index = (self.next_free_h_index + 1) % self.history_size
            return None



        self._sesop_grad_computations = 0
        def debug_step_callback(alpha):
            _loss, _grad = self.optimizer.loss_grad_func(alpha)
            s = session.run(self.summaries_in_iter, {self.optimizer._packed_loss_grad : _grad, self.new_loss : _loss})
            self.writer.add_summary(s, self.i_in_iter)
            self.i_in_iter += 1

        def debug_loss_callback(_loss, _grad):
            s = session.run(self.summaries_in_iter, {self.optimizer._packed_loss_grad : _grad, self.new_loss : _loss})
            self.writer.add_summary(s, self.i_in_iter)
            self.i_in_iter += 1
            self._sesop_grad_computations += 1


        #2 minimize by alpha
        self.optimizer.minimize(session=session, feed_dicts=feed_dicts, fetches=fetches, step_callback=None,
                                loss_callback=debug_loss_callback, override_loss_grad_func=override_loss_grad_func, additional_feed_dict=additional_feed_dict)

        additional_feed_dict.update({self.sesop_grad_computations: self._sesop_grad_computations})

        self.writer.add_summary(session.run(self.alpha_summary), self.i_after_iter)
        #seperate between different sesop runs
        for i in range(10):
            s = session.run(self.summaries_in_iter, {self.optimizer._packed_loss_grad : np.zeros(self.optimizer._packed_loss_grad.get_shape()), self.new_loss : 0})
            self.writer.add_summary(s, self.i_in_iter)
            self.i_in_iter += 1

        print 'loss after minimize = ' + str(avarge_on_feed_dicts(session, [self.new_loss], feed_dicts))

        #this is how much sesop + sgd moved
        _distance_sesop_moved = session.run(self.distance_sesop_moved)
        #TODO: remove additional_feed_dict
        self.writer.add_summary(session.run(self.distance_sesop_moved_summary, additional_feed_dict), self.i_after_iter)

        #3. move results into original variables:
        session.run(self.move_results_to_original)

        print 'loss after move_results_to_original = ' + str(avarge_on_feed_dicts(session, [self.orig_loss], feed_dicts))

        #this is how much sesop + sgd moved
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
        return _distance_seboost_moved
