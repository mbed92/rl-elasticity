import tensorflow as tf
import tensorflow.contrib as tfc


class Base(tf.keras.Model):

    @staticmethod
    def update(grads, network, optimizer):
        if grads:
            optimizer.apply_gradients(zip(grads, network.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())


class PolicyNetwork(Base):

    def __init__(self, num_controls):
        super(PolicyNetwork, self).__init__()
        self.num_controls = num_controls

        # self.rgb_process = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(16, (3, 3), 2, 'same', activation=None),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Conv2D(32, (3, 3), 4, 'same', activation=None),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=None),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Conv2D(128, (3, 3), 4, 'same', activation=None),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Conv2D(256, (3, 3), 2, 'same', activation=None),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(256, tf.nn.relu),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Dense(128, None)
        # ])

        self.state_estimator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(32, tf.nn.relu),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, None)
        ])

        self.LSTM = tf.keras.layers.LSTMCell(16)

        self.stddev_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(16, tf.nn.relu),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_controls, None)
        ])

        self.mean_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(16, tf.nn.relu),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_controls, None)
        ])

        self.hidden_state = None

    def call(self, inputs, training=None, mask=None):
        feed = tf.concat([tf.reshape(inputs[0], shape=[1, -1]), tf.reshape(inputs[1], shape=[1, -1])], 1)

        # rgb_logits = self.rgb_process(rgb, training=training)
        logits = self.state_estimator(feed, training=training)

        # push the hidden state into the LSTM
        # self.hidden_state = self.LSTM.get_initial_state(batch_size=tf.shape(logits)[0], dtype=logits.dtype) if self.hidden_state is None else self.hidden_state
        # logits, self.hidden_state = self.LSTM(logits, states=self.hidden_state, training=training)

        # estimate distributions of actions
        mu = tf.squeeze(self.mean_estimator(logits, training=training))
        log_std_dev = self.stddev_estimator(logits, training=training)
        sigma = tf.squeeze(tf.exp(log_std_dev))

        normal_dist = tf.distributions.Normal(mu, sigma)
        return normal_dist

    def compute_loss(self, action_distribution, action_samples, regularizer):
        policy_loss = action_distribution.log_prob(action_samples)
        policy_loss = tf.reduce_mean(policy_loss)

        # apply regularization
        reg_value = tfc.layers.apply_regularization(regularizer, self.trainable_variables)
        return policy_loss + reg_value


class ValueEstimator(Base):
    def __init__(self):
        super(ValueEstimator, self).__init__()
        self.value_estimator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(32, tf.nn.relu),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, None)
        ])


    def call(self, inputs, training=None, mask=None):
        feed = tf.concat([tf.reshape(inputs[0], shape=[1, -1]), tf.reshape(inputs[1], shape=[1, -1])], 1)
        logits = self.value_estimator(feed, training=training)
        return tf.squeeze(logits)

    def compute_loss(self, output, target, regularizer):
        value_loss = tf.losses.mean_squared_error(target, output)
        # reg_loss = tfc.layers.apply_regularization(regularizer, self.trainable_variables)
        return value_loss# + reg_loss
