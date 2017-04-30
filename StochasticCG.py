import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np


def _get_shape_tuple(tensor):
    return tuple(dim.value for dim in tensor.get_shape())


class StochasticCGOptimizer:
    # logit has to be softmax!
    # Every train var must be a scalar!
    def __init__(self, loss, train_variables, extra_train_ops, iters_count):
        self.loss = loss
        self.train_variables = train_variables
        self.extra_train_ops = extra_train_ops

        self.last_snapshot = [tf.Variable(initial_value=var.initialized_value(), trainable=False, dtype=tf.float32) for
                              var in
                              train_variables]
        self.conjugate = [
            tf.Variable(initial_value=tf.zeros(var.initialized_value().get_shape()), trainable=False, dtype=tf.float32)
            for var in
            train_variables]

        self.beta = tf.Variable(0.0)

        self.next_delta = None

        self.losses = []

        self.iters_count = iters_count  # The number of iters after which we update the conjugate vector

        self.init_training_op()

    def init_training_op(self):
        grads = tf.gradients(self.loss, self.train_variables)

        # We deduce from the gradient the current conjugate
        for i in range(len(grads)):
            grads[i] = grads[i] - self.beta * self.conjugate[i]

        self.grads = grads
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        apply_op = optimizer.apply_gradients(
            zip(grads, self.train_variables), name='train_step')
        self.train_ops = [apply_op]

    # user must call this when they have a session
    def init_ops(self, sess):
        self.sess = sess

        self.assign_into_conjugate_op_placeholder = [tf.placeholder(tf.float32) for x in self.train_variables]
        self.assign_into_conjugate_op = [tf.assign(x, y) for x, y in
                                         zip(self.conjugate, self.assign_into_conjugate_op_placeholder)]

        self.assign_into_beta_op_placeholder = tf.placeholder(tf.float32)
        self.assign_beta_op = tf.assign(self.beta, self.assign_into_beta_op_placeholder)

        self.assign_into_last_snapshot_op_placeholder = [tf.placeholder(tf.float32) for x in self.train_variables]
        self.assign_into_last_snapshot_op = [tf.assign(x, y) for x, y in
                                             zip(self.last_snapshot, self.assign_into_last_snapshot_op_placeholder)]

    def dump_debug(self, W, suffix):
        return
        with open('debug_' + suffix, 'w') as f:
            f.write('W = ' + str(W) + '\n')

    def mul_list(self, xs, ys):
        return [np.multiply(x, y) for x, y in zip(xs, ys)]

    def mul_list_by_scalar(self, xs, y):
        return [np.multiply(x, y) for x in xs]

    def div_list_by_scalar(self, xs, y):
        return [np.div(x, y) for x in xs]

    def div_list(self, xs, ys):
        return [np.divide(x, y) for x, y in zip(xs, ys)]

    def sub_list(self, xs, ys):
        return [np.subtract(x, y) for x, y in zip(xs, ys)]

    def add_list(self, xs, ys):
        return [np.add(x, y) for x, y in zip(xs, ys)]

    def sum_list(self, xs):
        return sum([np.sum(x) for x in xs])

    def sqrt_list(self, xs):
        return [np.sqrt(x) for x in xs]

    def norm_list(self, xs):
        return np.sqrt(self.sum_list(self.mul_list(xs, xs)))

    def approximate_next_delta(self, x, n, feed_dict):
        def sgd_iter(W):
            feed_dict = {}
            for var, w in zip(self.train_variables, W):
                feed_dict[var] = w

            grads = self.sess.run(self.grads, feed_dict=feed_dict)
            for i in range(len(grads)):
                W[i] -= grads[i] * 0.1

            return W

        # now run n regular sgd iterations
        W_0 = x

        self.dump_debug(W_0, 'before')

        W = [np.copy(w) for w in W_0]
        for i in range(n):
            W = sgd_iter(W)

        self.dump_debug(W_0, 'after')

        res = [w - w_0 for w, w_0 in zip(W, W_0)]
        self.dump_debug(res, 'approximate_next_delta')
        # now return the approximation
        return res

    # return beta
    def polak_riberie(self, x, delta, next_delta, feed_dict):


        #CG view:
        #delta = a*s_(n-1) is the result of the last line search
        #we calculate the gradient using n steps.
        #we would have want:
        #dir = real_next_delta + beta*delta
        #but it will go in direction  so we need to go in direction
        #dir = next_delta + beta*delta

        # we want not to go in direction delta anymore, and next_delta is prediction of where we would go
        # so we would have want to substract from real next_delta the component it has in direction delta
        # we return beta. To make sure we dont mess up the progress we did, we substract the direction of the previews step from the gradient
        # how much should we substract?
        ############### the whole component of the gradient in the direction of delta = <gradient, delta/|delta|>*delta ################
        # but we know in long term we will go to direction next_delta, and in real life we gradients are noisey
        # we could try approximate the general direction to which the gradients will point and use that as "how much should we substract":
        # The component of the gradient in the direction of next_delta
        #
        # gradient <-- gradient - <gradient, delta/|delta|>*delta
        # or
        # gradient <-- gradient - <gradient, next_delta=conjugate>

        # but next_delta points to the direction we will go, thus points to the direction that minimize the function,
        # thus -next_delta serves as an approximation to the direction of the gradient.
        # altough next_delta was calculated based on n steps, so its size is n*step_size*gradient_norm

        # but next_delta serves the use of the real gradient - a prediction to the direction minimizing the function

        # since conjugate <-- conjugate*beta + delta
        # which is what we will remove in the following iterations
        # the component of next_delta in the direction of delta is: dot(delta, next_delta/|next_delta|)

        #We remove the size in direction normalized_next_delta from the gradient
        # normalized_next_delta = self.div_list_by_scalar(next_delta, self.norm_list(next_delta))
        # dot = self.sum_list(self.mul_list(delta, normalized_next_delta))
        # return dot

        norm2 = self.norm_list(delta)
        norm1 = self.norm_list(next_delta)
        if norm1 == 0 or norm2 == 0:  # in that case, we want to return 0, so the conjugate get initlized by delta
            return 0

        beta = norm1/norm2
        return beta

        dot1 = self.sum_list(self.mul_list(next_delta, next_delta))
        dot2 = self.sum_list(self.mul_list(delta, delta))

        if dot2 == 0:  # in that case, we want to return 0, so the conjugate get initlized by delta
            return 0

        return dot1 / (dot2 * self.iters_count)

    def prepere_feed_dict_for_update(self, conjugate, beta, last_snapshot):

        feed_dict = {self.assign_into_beta_op_placeholder: beta}
        for i in range(len(self.train_variables)):
            feed_dict[self.assign_into_conjugate_op_placeholder[i]] = conjugate[i]
            feed_dict[self.assign_into_last_snapshot_op_placeholder[i]] = last_snapshot[i]

        return feed_dict

    def update_conjugate(self, feed_dict):

        # pull x, last_snapshot and conjugate
        x = self.sess.run(self.train_variables)
        last_snapshot = self.sess.run(self.last_snapshot)
        conjugate = self.sess.run(self.conjugate)

        # calc delta
        delta = self.sub_list(x, last_snapshot)


        next_delta = self.approximate_next_delta(x, self.iters_count, feed_dict)

        # calc beta
        beta = self.polak_riberie(x, delta, next_delta, feed_dict)

        # beta is expected to be around 1
        # So after k iterations, conjugate is expected to be delta*k norm
        # conjugate_new = conjugate*beta + delta

        # and we want ||conjugate_new|| = ||conjugate||

        # So we use conjugate_new2 = (||conjugate||/||conjugate_new||)*conjugate_new
        # then ||conjugate_new2|| = (||conjugate||/||conjugate_new||)*||conjugate_new|| = ||conjugate||
        print 'beta = ' + str(beta)
        print 'loss = ' + str(self.sess.run(self.loss, feed_dict=feed_dict))

        # if beta > 0.9:
        #     conjugate = [np.zeros(c.shape) for c in conjugate]

        # old_norm = self.sum_list(self.mul_list(delta, delta))
        # calc new conjugate
        conjugate = self.mul_list_by_scalar(conjugate, beta)

        # update conjugate, last_snapshot, beta
        feed_dict = self.prepere_feed_dict_for_update(conjugate=conjugate, beta=beta, last_snapshot=x)
        self.sess.run([self.assign_into_conjugate_op, self.assign_beta_op, self.assign_into_last_snapshot_op],
                      feed_dict=feed_dict)

    def run_iter(self, sess, feed_dict={}):
        if (len(self.losses) + 1) % self.iters_count == 0:
            print 'updating conjugate'
            self.update_conjugate(feed_dict)

        # print 'len(self.losses) = ' + str(len(self.losses))
        # _, losses, __ = sess.run([self.train_steps, self.losses, stages])

        sess.run(self.train_ops + self.extra_train_ops, feed_dict=feed_dict)
        loss = sess.run(self.loss, feed_dict=feed_dict)
        # print 'loss = ' + str(loss)
        self.losses.append(loss)
