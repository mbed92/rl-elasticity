import tensorflow as tf
import tensorflow.contrib as tfc

class Base(tf.keras.Model):

    @staticmethod
    def update(grads, network, optimizer):
        optimizer.apply_gradients(zip(grads, network.trainable_variables), global_step=tf.train.get_or_create_global_step())


class PolicyNetwork(Base):

    def __init__(self, num_controls):
        super(PolicyNetwork, self).__init__()
        self.num_controls = num_controls

        # self.state_estimator = tf.keras.Sequential([
        #     # tf.keras.layers.Dense(32, tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal()),
        #     tf.keras.layers.Dense(16, None, kernel_initializer=tf.keras.initializers.glorot_normal())
        # ])
        #
        # self.LSTM = tf.keras.layers.LSTMCell(16)

        self.stddev_estimator = tf.keras.Sequential([
            # tf.keras.layers.Dense(16, tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal()),
            tf.keras.layers.Dense(self.num_controls, None, dtype=tf.float64, input_shape=(400,), kernel_initializer=tf.keras.initializers.glorot_normal())
        ])

        self.mean_estimator = tf.keras.Sequential([
            # tf.keras.layers.Dense(16, tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal()),
            tf.keras.layers.Dense(self.num_controls, None, dtype=tf.float64, input_shape=(400,), kernel_initializer=tf.keras.initializers.glorot_normal())
        ])

        self.hidden_state = None

    def call(self, inputs, training=None, mask=None):
        # logits = self.state_estimator(inputs, training=training)

        # push the hidden state into the LSTM
        # self.hidden_state = self.LSTM.get_initial_state(batch_size=tf.shape(logits)[0], dtype=logits.dtype) if self.hidden_state is None else self.hidden_state
        # logits, self.hidden_state = self.LSTM(logits, states=self.hidden_state, training=training)

        # estimate distributions of actions
        inputs = tf.cast(inputs, dtype=tf.float64)
        mu = tf.squeeze(self.mean_estimator(inputs, training=training))
        log_std_dev = self.stddev_estimator(inputs, training=training)
        sigma = tf.squeeze(tf.exp(log_std_dev))

        normal_dist = tf.distributions.Normal(mu, sigma)
        return normal_dist

    def compute_loss(self, action_distribution, action_samples, advantage, regularizer=None):
        policy_loss = -action_distribution.log_prob(action_samples) * advantage
        policy_loss -= 2e-05 * action_distribution.entropy()
        policy_loss = tf.reduce_mean(policy_loss)

        if regularizer is not None:
            reg_value = tfc.layers.apply_regularization(regularizer, self.trainable_variables)
            policy_loss+=reg_value
        return policy_loss

class ValueEstimator(Base):
    def __init__(self):
        super(ValueEstimator, self).__init__()
        self.value_estimator = tf.keras.Sequential([
            # tf.keras.layers.Dense(64, tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal()),
            tf.keras.layers.Dense(1, None)
        ])


    def call(self, inputs, training=None, mask=None):
        logits = self.value_estimator(inputs, training=training)
        return tf.squeeze(logits)

    def compute_loss(self, value, target, regularizer=None):
        value_loss = tf.reduce_mean(tf.squared_difference(target, value))
        if regularizer is not None:
            reg_loss = tfc.layers.apply_regularization(regularizer, self.trainable_variables)
            value_loss+=reg_loss
        return value_loss
