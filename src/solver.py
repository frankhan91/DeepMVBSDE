import logging
import time
import numpy as np
import tensorflow as tf

DELTA_CLIP = 50.0


class SineBMSolver():
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        if self.net_config.loss_type == "DeepBSDE":
            self.model = SineBMNonsharedModel(config, bsde)
            self.opt_config = self.net_config.opt_config1
        elif self.net_config.loss_type == "DBDPsingle":
            self.model = SineBMNonsharedModelBDPSingle(config, bsde)
            self.opt_config = self.net_config.opt_config2
        self.y_init = self.model.y_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.opt_config.lr_boundaries, self.opt_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        mean_y_train = self.bsde.mean_y * 0

        # begin sgd iteration
        for step in range(self.opt_config.num_iterations+1):
            if self.eqn_config.type == 3 and step % 50 == 0:
                self.bsde.update_mean_y_estimate(mean_y_train)
                self.bsde.learn_drift()
            if step % self.opt_config.freq_resample == 0:
                train_data = self.bsde.sample(self.net_config.batch_size)
                train_data = (train_data[0], train_data[1], mean_y_train)
            mean_y_train = self.train_step(train_data)
            if step % self.net_config.logging_frequency == 0:
                loss, mean_y_valid = self.loss_fn(
                    (valid_data[0], valid_data[1], mean_y_train), training=False)
                loss = loss.numpy()
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    mean_y_valid = np.array([y.numpy() for y in mean_y_valid])
                    err_mean_y = np.mean((mean_y_valid - self.bsde.mean_y)**2)
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   err_mean_y: %.4e,    elapsed time: %3u" % (
                        step, loss, y_init, err_mean_y, elapsed_time))
        valid_data = self.bsde.sample(self.net_config.valid_size*20)
        _, mean_y_valid = self.loss_fn((valid_data[0], valid_data[1], mean_y_train), training=False)
        mean_y_valid = np.array([y.numpy() for y in mean_y_valid])
        print("Estimated mean_y:")
        print(self.bsde.mean_y)
        print("Error of mean_y:")
        print(mean_y_valid - self.bsde.mean_y)
        print("Average sqaured error of mean_y:")
        print(np.mean((mean_y_valid - self.bsde.mean_y)**2))
        train_result = {
            "history": np.array(training_history),
        }
        return train_result

    def loss_fn(self, inputs, training):
        dw, x, mean_y_input = inputs
        y_terminal, mean_y, loss_inter = self.model(inputs, training)
        y_target = self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        delta = y_terminal - y_target
        mean_y.append(tf.reduce_mean(y_target))
        # use linear approximation outside the clipped range
        loss = loss_inter + tf.reduce_mean(
            tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
            2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)
        )
        return loss, mean_y

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss, mean_y = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad, mean_y

    @tf.function
    def train_step(self, train_data):
        grad, mean_y = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return mean_y


class SineBMNonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[1]), name="yinit"
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim]), name="zinit"
                                  )
        self.subnet = [FeedForwardSubNet(config, self.eqn_config.dim) for _ in range(self.bsde.num_time_interval-1)]

    def call(self, inputs, training):
        loss_inter = 0
        mean_y = []
        dw, x, mean_y_input = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)
        mean_y.append(tf.reduce_mean(y))
        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            mean_y.append(tf.reduce_mean(y))
            if self.eqn_config.type == 2:
                y = y + (mean_y_input[t] - self.bsde.mean_y[t]) * self.bsde.delta_t
            z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        if self.eqn_config.type == 2:
            y = y + (mean_y_input[-2] - self.bsde.mean_y[-2]) * self.bsde.delta_t

        return y, mean_y, loss_inter


class SineBMNonsharedModelBDPSingle(tf.keras.Model):
    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[1]), name="yinit"
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim]), name="zinit"
                                  )
        self.subnetz = [FeedForwardSubNet(config, self.eqn_config.dim) for _ in range(self.bsde.num_time_interval-1)]
        self.subnety = [FeedForwardSubNet(config, 1) for _ in range(self.bsde.num_time_interval-1)]
        # self.nety = FeedForwardNoBNSubNet(config, 1)

    def call(self, inputs, training):
        loss_inter = 0
        mean_y = []
        dw, x, mean_y_input = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
        y_now = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)
        mean_y.append(tf.reduce_mean(y_now))
        for t in range(0, self.bsde.num_time_interval-1):
            y_predict = y_now - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y_now, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            if self.eqn_config.type == 2:
                y_predict = y_predict + (mean_y_input[t] - self.bsde.mean_y[t]) * self.bsde.delta_t
            y_next = self.subnety[t](x[:, :, t + 1], training)
            # y_next = self.nety(tf.concat([x[:, :, t+1], time_stamp[t+1]*all_one_vec], axis=-1))
            loss_inter += tf.reduce_mean((y_predict - tf.stop_gradient(y_next))**2)
            mean_y.append(tf.reduce_mean(y_next))
            y_now = y_next
            z = self.subnetz[t](x[:, :, t + 1], training) / self.bsde.dim
        # terminal time
        y_terminal = y_now - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y_now, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        if self.eqn_config.type == 2:
            y_terminal = y_terminal + (mean_y_input[-2] - self.bsde.mean_y[-2]) * self.bsde.delta_t

        return y_terminal, mean_y, loss_inter


class SineBMDBDPSolver():
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        self.nety_weights = [None] * self.bsde.num_time_interval # the first one will never be used
        self.netz_weights = [None] * self.bsde.num_time_interval # the first one will never be used
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[1]), name="yinit"
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim]), name="zinit"
                                  )
        self.netz = FeedForwardNoBNSubNet(config, self.eqn_config.dim)
        self.nety = FeedForwardNoBNSubNet(config, 1)
        self.nety_target = FeedForwardNoBNSubNet(config, 1)
        tmp_x = np.random.normal(size=[1, self.bsde.dim+1])
        self.nety(tmp_x)
        self.nety_target(tmp_x)
        self.netz(tmp_x)
        self.init_variables = [self.y_init, self.z_init]
        self.net_variables = self.nety.trainable_variables + self.netz.trainable_variables
        self.mean_y_train = self.bsde.mean_y * 0

        self.opt_config = self.net_config.opt_config3
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.opt_config.lr_boundaries, self.opt_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size, withtime=True)

        # begin sgd iteration
        for step in range(self.opt_config.num_sweep):
            if self.eqn_config.type == 3 and step % 1 == 0:
                print("Updating")
                self.bsde.update_mean_y_estimate(self.mean_y_train)
                self.bsde.learn_drift()
            if step % 1 == 0:
                train_data = self.bsde.sample(self.net_config.batch_size, withtime=True)
                train_data = (train_data[0], train_data[1], self.mean_y_train)
            self.train_one_sweep(train_data)
            if step % 1 == 0:
                loss, mean_y_valid = self.total_loss_fn(
                    (valid_data[0], valid_data[1], self.mean_y_train))
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    mean_y_valid = np.array([y.numpy() for y in mean_y_valid])
                    err_mean_y = np.mean((mean_y_valid - self.bsde.mean_y)**2)
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   err_mean_y: %.4e,    elapsed time: %3u" % (
                        step, loss, y_init, err_mean_y, elapsed_time))
        valid_data = self.bsde.sample(self.net_config.valid_size*20, withtime=True)
        _, mean_y_valid = self.total_loss_fn((valid_data[0], valid_data[1], self.mean_y_train))
        mean_y_valid = np.array([y.numpy() for y in mean_y_valid])
        print("Estimated mean_y:")
        print(self.bsde.mean_y)
        print("Error of mean_y:")
        print(mean_y_valid - self.bsde.mean_y)
        print("Average sqaured error of mean_y:")
        print(np.mean((mean_y_valid - self.bsde.mean_y)**2))
        train_result = {
            "history": np.array(training_history),
        }
        return train_result

    def train_one_sweep(self, train_data):
        if self.nety_weights[-1] is not None:
            self.nety.set_weights(self.nety_weights[-1])
            self.netz.set_weights(self.netz_weights[-1])
        for t in range(self.bsde.num_time_interval-1, -1, -1):
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))
            for _ in range(self.opt_config.num_iterations_perstep):
                if t > 0:
                    local_mean_y = self.train_step_inter(train_data, t)
                else:
                    local_mean_y = self.train_step_init(train_data)
            self.mean_y_train[t] = local_mean_y[0]
            if t == self.bsde.num_time_interval-1:
                self.mean_y_train[t+1] = local_mean_y[1]
            if t > 0:
                self.nety_weights[t] = self.nety.get_weights()
                self.netz_weights[t] = self.netz.get_weights()
                self.nety_target.set_weights(self.nety_weights[t])

    def local_loss_fn(self, inputs, t):
        dw, x, mean_y_input = inputs
        if t == self.bsde.num_time_interval-1:
            y_target = self.bsde.g_tf(self.bsde.total_time, x[:, :self.bsde.dim, -1])
            self.nety_target(x[:, :, t]) # to initialize variables
        else:
            y_target = self.nety_target(x[:, :, t+1])
        if t == 0:
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
            y = all_one_vec * self.y_init
            z = tf.matmul(all_one_vec, self.z_init)
        else:
            y = self.nety(x[:, :, t])
            z = self.netz(x[:, :, t]) / self.bsde.dim
        mean_y_estimate = [tf.reduce_mean(y), tf.reduce_mean(y_target)]
        y_next = y - self.bsde.delta_t * (
                self.bsde.f_tf(self.bsde.delta_t*t, x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
        if self.eqn_config.type == 2:
            y_next += (mean_y_input[t] - self.bsde.mean_y[t]) * self.bsde.delta_t
        delta = y_next - y_target
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(
            tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
            2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2)
        )
        return loss, mean_y_estimate

    def total_loss_fn(self, inputs):
        loss = 0
        mean_y = [None] * (self.bsde.num_time_interval+1)
        for t in range(self.bsde.num_time_interval-1, -1, -1):
            if t > 0:
                self.nety.set_weights(self.nety_weights[t])
                self.netz.set_weights(self.netz_weights[t])
            if t < self.bsde.num_time_interval-1:
                self.nety_target.set_weights(self.nety_weights[t+1])
            loss_tmp, mean_y_tmp = self.local_loss_fn(inputs, t)
            mean_y[t] = mean_y_tmp[0]
            if t == self.bsde.num_time_interval-1:
                mean_y[self.bsde.num_time_interval] = mean_y_tmp[1]
            loss += loss_tmp.numpy()
        return loss, mean_y

    def grad(self, inputs, t):
        with tf.GradientTape(persistent=True) as tape:
            loss, mean_y = self.local_loss_fn(inputs, t)
        if t == 0:
            grad = tape.gradient(loss, self.init_variables)
        else:
            grad = tape.gradient(loss, self.net_variables)
        del tape
        return grad, mean_y

    @tf.function
    def train_step_inter(self, train_data, t):
        grad, mean_y = self.grad(train_data, t)
        self.optimizer.apply_gradients(zip(grad, self.net_variables))
        return mean_y

    @tf.function
    def train_step_init(self, train_data):
        grad, mean_y = self.grad(train_data, 0)
        self.optimizer.apply_gradients(zip(grad, self.init_variables))
        return mean_y


class FlockSolver():
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        self.model = FlockNonsharedModel(config, bsde)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        valid_y2_init_true = self.bsde.y2_init_true_fn(valid_data["v_init"])

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if step % 50 == 0:
                simul_data = self.bsde.sample(self.net_config.simul_size)
                _, path_data = self.model.simulate_abstract(simul_data, False, "MC")
                self.bsde.learn_drift(path_data)
            train_data = self.bsde.sample(self.net_config.batch_size)
            self.train_step(train_data)
            if step % self.net_config.logging_frequency == 0:
                loss, _ = self.loss_fn(valid_data, training=False)
                loss = loss.numpy()
                elapsed_time = time.time() - start_time
                y2_init = self.model.y2_init_predict(valid_data).numpy()
                err_y2_init = np.mean((y2_init - valid_y2_init_true)**2)
                logging.info("step: %5u,    loss: %.4e, err_Y2_init: %.4e,    elapsed time: %3u" % (
                    step, loss, err_y2_init, elapsed_time))
                training_history.append([step, loss, err_y2_init, elapsed_time])
        np.random.seed(self.eqn_config.simul_seed)
        valid_data = self.bsde.sample(self.net_config.simul_size)
        valid_y2_init_true = self.bsde.y2_init_true_fn(valid_data["v_init"])
        y2_init = self.model.y2_init_predict(valid_data).numpy()
        print("Y2_true", valid_y2_init_true[:3])
        _, path_data = self.model.simulate_abstract(valid_data, training=False, drift_type="MC")
        print("Y2_approx", y2_init[:3])
        print("Std of v_terminal", path_data["v_std"].numpy()[-1])
        y2_err = np.mean((y2_init - valid_y2_init_true)**2)
        y2_square = np.mean(y2_init**2)
        train_result = {
            "history": np.array(training_history),
            "y2_err": y2_err,
            "R2": 1 - y2_err / y2_square,
            "v_std": path_data["v_std"].numpy()
        }
        return train_result

    def loss_fn(self, inputs, training):
        y_terminal, path_data = self.model(inputs, training)
        delta = y_terminal
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss, path_data

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss, path_data = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad, path_data

    @tf.function
    def train_step(self, train_data):
        grad, path_data = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return path_data


class FlockNonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y_init_net = FeedForwardSubNet(config, self.eqn_config.dim*2)
        self.z_subnet = [FeedForwardSubNet(config, self.eqn_config.dim**2*2) for _ in range(self.bsde.num_time_interval)]
        # self.y2_init_net = FeedForwardSubNet(config, self.eqn_config.dim)
        # self.z_subnet = [FeedForwardSubNet(config, self.eqn_config.dim**2) for _ in range(self.bsde.num_time_interval)]

    def simulate_abstract(self, inputs, training, drift_type):
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(inputs["dw"])[0], 1]), dtype=self.net_config.dtype)
        x, v = inputs["x_init"], inputs["v_init"]
        v_std = [tf.math.reduce_std(v[:, 0])]
        y = self.y_init_net(tf.concat([x, v], axis=1))
        y1, y2 = y[:, :self.eqn_config.dim], y[:, self.eqn_config.dim:]
        z = self.z_subnet[0](v, training) / self.bsde.dim
        z = tf.reshape(z, [-1, 2*self.eqn_config.dim, self.eqn_config.dim])
        drift_input = tf.zeros(shape=[0, 2*self.eqn_config.dim+1], dtype="float64")
        y_drift_label = tf.zeros(shape=[0, 2*self.eqn_config.dim], dtype="float64")
        for t in range(0, self.bsde.num_time_interval):
            t_input = t*self.bsde.delta_t*all_one_vec
            if drift_type == "NN":
                y1_drift_term, y2_drift_term = self.bsde.y_drift_nn(t_input, x, v)
            elif drift_type == "MC":
                y1_drift_term, y2_drift_term = self.bsde.y_drift_mc(t_input, x, v)
                y_drift_term = tf.concat([y1_drift_term, y2_drift_term], axis=1)
                y_drift_label = tf.concat([y_drift_label, y_drift_term], axis=0)
                drift_input = tf.concat([drift_input, tf.concat([t_input, x, v], axis=1)], axis=0)
            x = x + v * self.bsde.delta_t
            v = v - y2 / self.bsde.R / 2 * self.bsde.delta_t + self.bsde.C * inputs["dw"][:, :, t]
            diffusion_term = (z @ inputs["dw"][:, :, t:t+1])[..., 0]
            y1 = y1 - (y1_drift_term) * self.bsde.delta_t + diffusion_term[:, :self.eqn_config.dim]
            y2 = y2 - (y1 + y2_drift_term) * self.bsde.delta_t + diffusion_term[:, self.eqn_config.dim:]
            v_std.append(tf.math.reduce_std(v[:, 0]))
            if t < self.bsde.num_time_interval-1:
                z = self.z_subnet[t+1](v, training) / self.bsde.dim
                z = tf.reshape(z, [-1, 2*self.eqn_config.dim, self.eqn_config.dim])
        v_std = tf.stack(v_std, axis=0)
        path_data = {"input": drift_input, "y_drift": y_drift_label, "v_std": v_std}
        y = tf.concat([y1, y2], axis=1)
        return y, path_data

    # def simulate_y2_abstract(self, inputs, training, drift_type):
    #     all_one_vec = tf.ones(shape=tf.stack([tf.shape(inputs["dw"])[0], 1]), dtype=self.net_config.dtype)
    #     v = inputs["v_init"]
    #     y2 = self.y2_init_net(v)
    #     # y2 = self.y2_init_net(v) * 0 + self.bsde.y2_init_true_fn(v)
    #     z = self.z_subnet[0](v, training) / self.bsde.dim
    #     z = tf.reshape(z, [-1, self.eqn_config.dim, self.eqn_config.dim])
    #     # z = tf.reshape(z, [-1, self.eqn_config.dim, self.eqn_config.dim]) * 0 + self.bsde.eta[0] * self.bsde.C
    #     drift_input = tf.zeros(shape=[0, self.eqn_config.dim+1], dtype="float64")
    #     y2_drift_label = tf.zeros(shape=[0, self.eqn_config.dim], dtype="float64")
    #     for t in range(0, self.bsde.num_time_interval):
    #         t_input = t*self.bsde.delta_t*all_one_vec
    #         if drift_type == "NN":
    #             y2_drift_term = self.bsde.y2_drift_nn(v, t_input)
    #         elif drift_type == "MC":
    #             y2_drift_term = self.bsde.y2_drift_mc(v, t_input)
    #             y2_drift_label = tf.concat([y2_drift_label, y2_drift_term], axis=0)
    #             drift_input = tf.concat([drift_input, tf.concat([v, t_input], axis=-1)], axis=0)
    #         v =  v - y2 / self.bsde.R / 2 * self.bsde.delta_t + self.bsde.C * inputs["dw"][:, :, t]
    #         y2 = y2 - 2 * (y2_drift_term) * self.bsde.Q * self.bsde.delta_t + (z @ inputs["dw"][:, :, t:t+1])[..., 0]
    #         if t < self.bsde.num_time_interval-1:
    #             z = self.z_subnet[t+1](v, training) / self.bsde.dim
    #             z = tf.reshape(z, [-1, self.eqn_config.dim, self.eqn_config.dim])
    #             # z = tf.reshape(z, [-1, self.eqn_config.dim, self.eqn_config.dim]) * 0 + self.bsde.eta[t+1] * self.bsde.C
    #     path_data = {"input": drift_input, "y_drift": y2_drift_label, "v_terminal": v}
    #     return y2, path_data

    def y2_init_predict(self, inputs, training=False):
        return self.y_init_net(tf.concat([inputs["x_init"], inputs["v_init"]], axis=1), training)[:, self.eqn_config.dim:]
        # return self.y2_init_net(inputs["v_init"], training)

    def call(self, inputs, training):
        return self.simulate_abstract(inputs, training, "NN")


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config, dim_out):
        super().__init__()
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim_out, activation=None))

    def __call__(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        return x


class FeedForwardNoBNSubNet(tf.keras.Model):
    def __init__(self, config, dim_out):
        super().__init__()
        num_hiddens = config.net_config.num_hiddens
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim_out, activation=None))

    def __call__(self, x):
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        return x
