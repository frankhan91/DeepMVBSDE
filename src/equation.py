import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import multivariate_normal as normal
import timeit

class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.t_grid = np.linspace(0, self.total_time, self.num_time_interval+1)
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class SineBM(Equation):
    """Sine of BM in the note"""
    def __init__(self, eqn_config):
        super(SineBM, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.mean_y = np.sin(self.t_grid)*np.exp(-self.t_grid/2)
        self.drift_model = self.create_model()
        self.learn_forward()

    def create_model(self):
        net_width = [16, 16]
        activation = 'relu'
        inputs = keras.Input(shape=(self.dim+1,))
        x = keras.layers.Dense(net_width[0], activation=activation)(inputs)
        for w in net_width[1:]:
            x = keras.layers.Dense(w, activation=activation)(x)
        outputs = keras.layers.Dense(1, activation='relu')(x)
        return keras.Model(inputs, outputs)

    def learn_forward(self):
        N_simu = 500
        N_learn = 500
        N_iter = 3
        dim = self.dim
        Nt = self.num_time_interval
        dt = self.delta_t

        batch_size = 128
        epochs = 80            

        for i in range(N_iter):
            x_path = np.zeros(shape=[N_simu, Nt+1, dim])

            for i, t in enumerate(self.t_grid):
                drift_true = np.exp(-np.sum(x_path[:, i]**2, axis=-1, keepdims=True)/(dim + 2*t))*(dim/(dim+2*t))**(dim/2)
                t_tmp = np.zeros(shape=[N_simu, 1]) + t
                x_tmp = np.concatenate([t_tmp, x_path[:, i]], axis=-1)
                drift_nn = self.drift_model.predict(x_tmp)
                if i < Nt:
                    x_path[:, i+1, :] = x_path[:, i, :] + np.sin(drift_nn - drift_true) * dt + \
                        np.random.normal(scale=np.sqrt(dt), size=(N_simu, dim))

            term_approx = np.zeros(shape=[N_learn, self.t_grid.shape[0]])
            term_true = np.zeros(shape=[N_learn, self.t_grid.shape[0]])
            path_idx = np.random.choice(N_simu, N_learn, replace=False)
            for i, t in enumerate(self.t_grid):
                term_true[:, i] = np.exp(-np.sum(x_path[path_idx, i]**2, axis=-1)/(dim + 2*t))*(dim/(dim+2*t))**(dim/2)
                diff_x = x_path[path_idx, None, i, :] - x_path[None, path_idx, i, :]
                norm = np.sum(diff_x**2, axis=-1)
                term_approx[:, i] = np.average(np.exp(-norm / dim), axis=1)
            # learn drift model
            self.drift_model = self.create_model()
            t_tmp = self.t_grid[None, :, None] + np.zeros(shape=[N_learn, Nt+1, 1])
            X = np.concatenate([t_tmp, x_path[path_idx]], axis=-1)
            X = X.reshape([-1, dim+1])
            Y = term_approx.reshape([-1, 1])
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                0.001, decay_steps=200, decay_rate=0.9
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            self.drift_model.compile(loss='mse',optimizer=optimizer)
            hist = self.drift_model.fit(
                X, Y, batch_size=batch_size, 
                epochs=epochs, verbose=0,
                validation_split=0.05
            )
            Y_predict = self.drift_model.predict(X)
            Y_true = term_true.reshape([-1, 1])
            r2 = np.sum((Y_predict - Y_true)**2)/np.sum((Y_true-np.mean(Y_true))**2)
            print("R^2: {}".format(r2))
        assert r2 < 0.01, "Failed learning of the forward model"

    def sample(self, num_sample):
        # start = timeit.default_timer()
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        dim = self.dim
        Nt = self.num_time_interval
        dt = self.delta_t
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, self.dim])
        t_tmp = np.zeros(shape=[num_sample, 1])
        for i, t in enumerate(self.t_grid):
            drift_true = np.exp(-np.sum(x_sample[:, i]**2, axis=-1, keepdims=True)/(dim + 2*t))*(dim/(dim+2*t))**(dim/2)
            x_tmp = np.concatenate([t_tmp, x_sample[:, i]], axis=-1)
            drift_nn = self.drift_model.predict(x_tmp)
            t_tmp += self.delta_t
            if i < self.num_time_interval:
                x_sample[:, i+1, :] = x_sample[:, i, :] + np.sin(drift_nn - drift_true) * dt + \
                    dw_sample[:, :, i]
        x_sample = x_sample.transpose((0, 2, 1))
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)  
        return dw_sample, x_sample

        # x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        # x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        # for i in range(self.num_time_interval):
        #     x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        # return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        term1 = tf.reduce_sum(z, 1, keepdims=True)/np.sqrt(self.dim) - y/2
        term2 = tf.sqrt(1 + y**2 + tf.reduce_sum(z**2, 1, keepdims=True)) - np.sqrt(2)
        return  - term1 - term2

    def g_tf(self, t, x):
        return tf.math.sin((self.total_time + tf.reduce_sum(x, 1, keepdims=True)/np.sqrt(self.dim)))
