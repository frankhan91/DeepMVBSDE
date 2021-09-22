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

        self.model = SineBMNonsharedModel(config, bsde)
        self.y_init = self.model.y_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        mean_y_train = self.bsde.mean_y * 0

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if self.eqn_config.type == 3 and step % 50 == 0:
                self.bsde.update_mean_y_estimate(mean_y_train)
                self.bsde.learn_drift()
            if step % 20 == 0:
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
        print(self.bsde.mean_y)
        print(mean_y_valid - self.bsde.mean_y)
        print(np.mean((mean_y_valid - self.bsde.mean_y)**2))
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x, mean_y_input = inputs
        y_terminal, mean_y = self.model(inputs, training)
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

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
                                                    size=[1])
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim])
                                  )
        self.subnet = [FeedForwardSubNet(config, self.eqn_config.dim) for _ in range(self.bsde.num_time_interval-1)]

    def call(self, inputs, training):
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
        mean_y.append(tf.reduce_mean(y))

        return y, mean_y


class FlockSolver():
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        self.model = FlockSimpleNonsharedModel(config, bsde)
        self.y2_init = self.model.y2_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        mean_v_input = np.zeros([1, self.bsde.dim, self.bsde.num_time_interval+1])

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            train_data = self.bsde.sample(self.net_config.batch_size)
            train_data["mean_v_input"] = mean_v_input
            path_data = self.train_step(train_data)
            mean_v_input = mean_v_input * 0.95 + path_data["mean_v"].numpy() * 0.05
            if step % self.net_config.logging_frequency == 0:
                valid_data["mean_v_input"] = mean_v_input
                loss, _ = self.loss_fn(valid_data, training=False)
                loss = loss.numpy()
                y2_init = self.y2_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y2_init, elapsed_time])
                if self.net_config.verbose:
                    err_y2_init = np.mean((y2_init - self.bsde.y2_init_true)**2)
                    err_mean_v = np.mean((mean_v_input - self.bsde.v_init[..., None])**2)
                    logging.info("step: %5u,    loss: %.4e, err_Y2_init: %.4e,   err_mean_v: %.4e,    elapsed time: %3u" % (
                        step, loss, err_y2_init, err_mean_v, elapsed_time))
        print(self.bsde.y2_init_true)
        print(y2_init)
        print(self.bsde.v_init)
        print(mean_v_input[0, :, -1])
        return np.array(training_history)

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


class FlockSimpleNonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y2_init = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[self.eqn_config.dim])
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim**2])
                                  )
        self.subnet = [FeedForwardSubNet(config, self.eqn_config.dim**2) for _ in range(self.bsde.num_time_interval-1)]

    def call(self, inputs, training):
        mean_v = []
        mean_v_input = inputs["mean_v_input"]
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(inputs["dw"])[0], 1]), dtype=self.net_config.dtype)
        y2 = all_one_vec * self.y2_init
        z = tf.matmul(all_one_vec, self.z_init)
        z = tf.reshape(z, [-1, self.eqn_config.dim, self.eqn_config.dim])
        v = inputs["v_init"]
        mean_v.append(tf.reduce_mean(v, axis=0, keepdims=True))
        for t in range(0, self.bsde.num_time_interval):
            v =  v - y2 / self.bsde.R / 2 * self.bsde.delta_t + self.bsde.C * inputs["dw"][:, :, t]
            mean_v.append(tf.reduce_mean(v, axis=0, keepdims=True))
            y2 = y2 - 2 * (v - mean_v_input[..., t]) * self.bsde.Q * self.bsde.delta_t + (z @ inputs["dw"][:, :, t:t+1])[..., 0]
            if t < self.bsde.num_time_interval-1:
                z = self.subnet[t](v, training) / self.bsde.dim
                z = tf.reshape(z, [-1, self.eqn_config.dim, self.eqn_config.dim])
        mean_v = tf.stack(mean_v, axis=-1)
        path_data = {"mean_v": mean_v}
        return y2, path_data


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
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim_out, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        return x
