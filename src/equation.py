import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.integrate import solve_ivp

class Equation():
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        self.eqn_config = eqn_config
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
        self.mean_y_estimate = self.mean_y * 0
        self.drift_model = self.create_model()
        if eqn_config.type != 3:
            self.learn_drift()

    def create_model(self):
        num_hiddens = self.eqn_config.num_hiddens
        activation = 'softplus'
        inputs = keras.Input(shape=(self.dim+1,))
        x = keras.layers.Dense(num_hiddens[0], activation=activation)(inputs)
        for w in num_hiddens[1:]:
            x = keras.layers.Dense(w, activation=activation)(x)
        outputs = keras.layers.Dense(1, activation='softplus')(x)
        return keras.Model(inputs, outputs)

    def learn_drift(self):
        N_simu = self.eqn_config.N_simu
        N_learn = self.eqn_config.N_learn
        N_iter = 3
        dim = self.dim
        Nt = self.num_time_interval
        dt = self.delta_t

        batch_size = 128
        epochs = 80

        for _ in range(N_iter):
            x_path = np.zeros(shape=[N_simu, Nt+1, dim])

            for i, t in enumerate(self.t_grid):
                drift_true = np.exp(-np.sum(x_path[:, i]**2, axis=-1, keepdims=True)/(dim + 2*t))*(dim/(dim+2*t))**(dim/2)
                t_tmp = np.zeros(shape=[N_simu, 1]) + t
                x_tmp = np.concatenate([t_tmp, x_path[:, i]], axis=-1)
                drift_nn = self.drift_model.predict(x_tmp)
                if i < Nt:
                    x_path[:, i+1, :] = x_path[:, i, :] + np.sin(drift_nn - drift_true) * dt + \
                        np.random.normal(scale=np.sqrt(dt), size=(N_simu, dim))
                    if self.eqn_config.type == 3:
                        x_path[:, i+1, :] += self.eqn_config.couple_coeff * (self.mean_y_estimate[i] - self.mean_y[i]) * dt

            term_approx = np.zeros(shape=[N_learn, self.t_grid.shape[0]]) # pylint: disable=unsubscriptable-object
            term_true = np.zeros(shape=[N_learn, self.t_grid.shape[0]]) # pylint: disable=unsubscriptable-object
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
        # assert r2 < 0.01, "Failed learning of the forward model"

    def sample(self, num_sample, withtime=False):
        # start = timeit.default_timer()
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
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
                if self.eqn_config.type == 3:
                    x_sample[:, i+1, :] += self.eqn_config.couple_coeff * (self.mean_y_estimate[i] - self.mean_y[i]) * dt
        if withtime:
            t_data = np.zeros([num_sample, self.num_time_interval + 1, 1])
            for i, t in enumerate(self.t_grid):
                t_data[:, i, :] = t
            x_sample = np.concatenate([x_sample, t_data], axis=-1)
        x_sample = x_sample.transpose((0, 2, 1))
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)
        return dw_sample, x_sample

        # x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        # x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        # for i in range(self.num_time_interval):
        #     x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        # return dw_sample, x_sample

    def update_mean_y_estimate(self, mean_y_estimate):
        self.mean_y_estimate = mean_y_estimate.copy()

    def f_tf(self, t, x, y, z):
        # should not depend on x
        term1 = tf.reduce_sum(z, 1, keepdims=True)/np.sqrt(self.dim) - y/2
        term2 = tf.sqrt(1 + y**2 + tf.reduce_sum(z**2, 1, keepdims=True)) - np.sqrt(2)
        return  - term1 - term2

    def g_tf(self, t, x):
        return tf.math.sin(t + tf.reduce_sum(x, 1, keepdims=True)/np.sqrt(self.dim))

    def z_true(self, t, x):
        return tf.math.cos(t + tf.reduce_sum(x, 1, keepdims=True)/np.sqrt(self.dim)) / np.sqrt(self.dim)


class Flocking(Equation):
    """Flocking in the note"""
    def __init__(self, eqn_config):
        super(Flocking, self).__init__(eqn_config)
        self.x_init_mean = 0
        self.x_init_sigma = 1.0
        self.v_init_mean = 1
        self.v_init_sigma = 1.0
        self.R, self.Q, self.C = 0.5, 1, 0.1
        self.eta, self.xi = self.riccati_solu()
        # self.y2_init_true = self.eta[0] @ self.v_init + self.xi[0]
        self.y2_init_true_fn = lambda v: v @ self.eta[0].transpose() + self.xi[0][None, :]
        self.y_drift_model = self.create_model()

    def riccati_solu(self):
        n = self.dim
        def riccati(t, y):
            eta = np.reshape(y[:n**2], (n, n))
            xi = y[n**2:]
            deta = 2 * self.Q * np.identity(n) - eta @ eta / self.R / 2
            dxi = -2 * self.Q * self.v_init_mean - eta @ xi / self.R / 2
            dy = np.concatenate([deta.reshape([-1]), dxi])
            return dy

        y_init = np.zeros([n**2+n])
        sol = solve_ivp(riccati, [0, self.total_time], y_init,
                        t_eval=np.linspace(0, self.total_time, self.num_time_interval+1))
        sol_path = np.flip(sol.y, axis=-1)
        eta_path = np.reshape(sol_path[:n**2].transpose(), (self.num_time_interval+1, n, n))
        xi_path = sol_path[n**2:].transpose()
        return eta_path, xi_path

    def sample(self, num_sample):
        dw_sample = np.random.normal(scale=self.sqrt_delta_t, size=[num_sample, self.dim, self.num_time_interval])
        x_init = np.random.normal(loc=self.x_init_mean, scale=self.x_init_sigma, size=[num_sample, self.dim])
        v_init = np.random.normal(loc=self.v_init_mean, scale=self.v_init_sigma, size=[num_sample, self.dim])
        data_dict = {"dw": dw_sample, "x_init": x_init, "v_init": v_init}
        return data_dict

    def create_model(self):
        net_width = [48, 48]
        activation = 'relu'
        # inputs = keras.Input(shape=(self.dim+1,))
        inputs = keras.Input(shape=(2*self.dim+1,))
        x = keras.layers.Dense(net_width[0], activation=activation, dtype="float64")(inputs)
        for w in net_width[1:]:
            x = keras.layers.Dense(w, activation=activation, dtype="float64")(x)
        # outputs = keras.layers.Dense(self.dim, activation=None, dtype="float64")(x)
        outputs = keras.layers.Dense(2*self.dim, activation=None, dtype="float64")(x)
        return keras.Model(inputs, outputs)

    def learn_drift(self, path_data):
        batch_size = 128
        epochs = 3
        # learn drift model
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=200, decay_rate=1
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.y_drift_model.compile(loss='mse',optimizer=optimizer)
        X, Y = path_data["input"].numpy(), path_data["y_drift"].numpy()
        hist = self.y_drift_model.fit(
            X, Y, batch_size=batch_size,
            epochs=epochs, verbose=0,
            validation_split=0.05
        )

    def y_drift_nn(self, t, x, v):
        y_drift = self.y_drift_model(tf.concat([t, x, v], axis=1))
        y1_drift, y2_drift = y_drift[:, :self.dim], y_drift[:, self.dim:]
        return y1_drift, y2_drift

    def y_drift_mc(self, t, x, v):
        delta_x = tf.expand_dims(x, axis=1) - x # B * B * 3, x - x'
        delta_v = tf.expand_dims(v, axis=1) - v # B * B * 3. v - v'
        x_norm2 = tf.reduce_sum(delta_x**2, axis=2, keepdims=True) # B * B * 1
        weight = 1 / tf.math.pow(1+x_norm2, self.eqn_config.beta) # B * B * 1
        weight_mean = tf.reduce_mean(weight, axis=1) # B * 1
        partial_weight = -2 * self.eqn_config.beta * tf.expand_dims(x, axis=1) / tf.math.pow(1+x_norm2, self.eqn_config.beta+1) # B * B * 3
        partial_weight_v = tf.expand_dims(partial_weight, axis=-1) @ tf.expand_dims(-delta_v, axis=2) # B * B * 3 * 3
        partial_weight_v_mean = tf.reduce_mean(partial_weight_v, axis=1) # B * 3 * 3
        weight_v_mean = tf.reduce_mean(weight * -delta_v, axis=1) # B * 3
        weight_v_mean_expand = tf.expand_dims(weight_v_mean, axis=-1) # B * 3 * 1
        y1_drift = 2 * self.Q * partial_weight_v_mean @ weight_v_mean_expand
        y1_drift = y1_drift[..., 0]
        y2_drift = -2 * self.Q * weight_mean * weight_v_mean
        return y1_drift, y2_drift

    # def y2_drift_nn(self, v, t):
    #     return self.y_drift_model(tf.concat([v, t], axis=-1))

    # def y2_drift_mc(self, v, t):
    #     return v - tf.stop_gradient(tf.reduce_mean(v, axis=0, keepdims=True))
