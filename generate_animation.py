from tensorflow.python.summary import event_accumulator as ea
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

from lr_auto_adjust_sgd_optimizer import SgdAdjustOptimizer

config = tf.ConfigProto(
        device_count={'GPU': 0}
)
sess = tf.Session(config=config)


class TrajAnimator:
    def __init__(self, **optimizer_kwargs):
        self.optimizer_kwargs = optimizer_kwargs


    def create_objective_function(self, noise_power, batch_size):
        dim = 2
        x = tf.Variable(np.ones([dim, 1]), name='x', dtype=tf.float32)
        A_golden = tf.Variable(np.asarray([[8, 0],
                                    [0, 1]]), name='A_golden', trainable=False, dtype=tf.float32)

        A = tf.Variable(np.asarray([[8, 0],
                                    [0, 1]]), name='A', trainable=False, dtype=tf.float32)

        rs = tf.random_uniform(A.get_shape().as_list() + [batch_size], minval=-noise_power, maxval=noise_power)
        r = tf.reduce_mean(rs, axis=2)


        noise_A_op = tf.assign(A, A_golden + r)

        Ax = tf.matmul(A, x)
        xAx = tf.matmul(tf.transpose(x), Ax)

        return x, A, xAx, noise_A_op


    def simulate_noisey_trajectory(self, sess, noise_power, batch_size, n):
        x, A, xAx, noise_A_op = self.create_objective_function(noise_power, batch_size)

        #create an optimizer op:
        if self.optimizer_kwargs['optimizer_name'] == 'SGD':
            minimize_op = tf.train.GradientDescentOptimizer(self.optimizer_kwargs['lr'])
        elif self.optimizer_kwargs['optimizer_name'] == 'Adam':
            minimize_op = tf.train.AdamOptimizer(self.optimizer_kwargs['lr'], self.optimizer_kwargs['beta1'], self.optimizer_kwargs['beta2'])
        elif self.optimizer_kwargs['optimizer_name'] == 'Momentum':
            minimize_op = tf.train.MomentumOptimizer(self.optimizer_kwargs['lr'], self.optimizer_kwargs['momentum'], use_nesterov=False)
        elif self.optimizer_kwargs['optimizer_name'] == 'Nestrov':
            minimize_op = tf.train.MomentumOptimizer(self.optimizer_kwargs['lr'], self.optimizer_kwargs['momentum'], use_nesterov=True)
        elif self.optimizer_kwargs['optimizer_name'] == 'Adagrad':
            minimize_op = tf.train.AdagradOptimizer(self.optimizer_kwargs['lr'])
        elif self.optimizer_kwargs['optimizer_name'] == 'Adadelta':
            minimize_op = tf.train.AdadeltaOptimizer(self.optimizer_kwargs['lr'], self.optimizer_kwargs['rho'])
        elif self.optimizer_kwargs['optimizer_name'] == 'Adam-Adjust':
            self.optimizer_kwargs['use_locking'], self.optimizer_kwargs['name'] = False, 'SgdAdjustOptimizer'
            minimize_op = SgdAdjustOptimizer(tf.trainable_variables(), **self.optimizer_kwargs)
        else:
            assert (False)

        minimize_op = minimize_op.minimize(xAx)
        #init
        sess.run(tf.global_variables_initializer())

        #set starting point
        x0 = np.asarray([1, -1]).reshape(2,1)
        sess.run(tf.assign(x, x0))

        #create meshgrid
        X = np.arange(-1.5, 1.5, 0.1)
        Y = np.arange(-1.5, 1.5, 0.1)
        X, Y = np.meshgrid(X, Y)


        #compute trajectory
        traj = [x0]
        traj_f = [sess.run(xAx, {x : x0 })[0][0]]

        Zs = []
        for i in range(n):

            #get the objective surface for current noise
            @np.vectorize
            def f(_x, _y):
                return sess.run(xAx, {x : np.asarray([_x, _y]).reshape([2, 1]) })[0][0]
            Z = f(X,Y)
            #print Z.shape
            Zs.append(Z)


            #run minimization step on that surface
            sess.run(minimize_op)

            #move to next noise
            sess.run(noise_A_op)

            traj.append(sess.run(x))
            traj_f.append(sess.run(xAx))

        return traj, Zs, traj_f




    def animate_sgd_trajectory(self, sess, noise_power, batch_size, n):
        traj, Zs, traj_f = self.simulate_noisey_trajectory(sess, noise_power, batch_size, n)

        traj_x = [point[0] for point in traj]
        traj_y = [point[1] for point in traj]

        #create meshgrid
        X = np.arange(-1.5, 1.5, 0.1)
        Y = np.arange(-1.5, 1.5, 0.1)
        X, Y = np.meshgrid(X, Y)

        plt.xkcd()
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.view_init(30, 90)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.title(self.optimizer_kwargs['optimizer_name'] + ' trajectory')
        surf = ax.plot_surface(X, Y, Zs[0], rstride=1, cstride=1, antialiased=True)

        def update_surface_and_traj(i, data, surf):
            print 'i = ' + str(i)
            ax.clear()
            plot = ax.plot_surface(X, Y, Zs[i])

            ax.plot(traj_x[:i], traj_y[:i], '-', zs=traj_f[:i], color='r')


            return plot,

        surf_ani = animation.FuncAnimation(fig, update_surface_and_traj, n, fargs=(Zs, surf),
                                       interval=self.optimizer_kwargs['interval'], blit=True)

        surf_ani.save('/home/shai/animations/' + self.optimizer_kwargs['optimizer_name'] + '_anim.mp4')

noise_power = 0.5
n = 100

# TrajAnimator(optimizer_name='SGD', lr=0.1).animate_sgd_trajectory(sess, noise_power=noise_power, batch_size=1, n=n)
# TrajAnimator(optimizer_name='Momentum', lr=0.01, momentum=0.9).animate_sgd_trajectory(sess, noise_power=noise_power, batch_size=1, n=n)
# TrajAnimator(optimizer_name='Adam', lr=0.1, beta1=0.9, beta2=0.999).animate_sgd_trajectory(sess, noise_power=noise_power, batch_size=1, n=n)
# TrajAnimator(optimizer_name='Nestrov', lr=0.01, momentum=0.9).animate_sgd_trajectory(sess, noise_power=noise_power, batch_size=1, n=n)

TrajAnimator(optimizer_name='Adam', lr=0.001, beta1=0.9, beta2=0.999, interval=100).animate_sgd_trajectory(sess, noise_power=noise_power, batch_size=1, n=n)
# TrajAnimator(optimizer_name='Adam-Adjust',
#              lr=0.001,
#              learning_rate=0.001,
#              iters_per_adjust=5,
#              update_rule='linear',
#              per_variable=False,
#              ignore_big_ones=False,
#              base_optimizer='Adam',
#              beta1=0.9,
#              beta2=0.999,
#              interval=100).animate_sgd_trajectory(sess, noise_power=noise_power, batch_size=1, n=n)