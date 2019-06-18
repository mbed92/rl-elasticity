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

        self.pose_process = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, None)
        ])

        self.joints_process = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, None)
        ])

        self.LSTM = tf.keras.layers.LSTMCell(64)

        self.stddev_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_controls, None)
        ])

        self.mean_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_controls, None)
        ])

        self.hidden_state = None

    def call(self, inputs, training=None, mask=None):
        # rgb = inputs[0]
        poses = inputs[0]
        joints = inputs[1]

        # rgb_logits = self.rgb_process(rgb, training=training)
        pos_logits = self.pose_process(poses, training=training)
        joi_logits = self.joints_process(joints, training=training)

        state = tf.concat([pos_logits, joi_logits], axis=0)
        integrator_feed = tf.reduce_mean(state, axis=0, keepdims=True)

        # push the hidden state into the LSTM
        self.hidden_state = self.LSTM.get_initial_state(batch_size=tf.shape(integrator_feed)[0], dtype=integrator_feed.dtype) if self.hidden_state is None else self.hidden_state
        logits, self.hidden_state = self.LSTM(integrator_feed, states=self.hidden_state, training=training)

        # estimate distributions of actions
        mu = tf.squeeze(self.mean_estimator(logits, training=training))
        sigma = tf.squeeze(tf.nn.softplus(self.stddev_estimator(logits, training=training)) + 1e-05)
        normal_dist = tf.distributions.Normal(mu, sigma)

        return normal_dist

    def compute_loss(self, action_distribution, action_samples, regularizer, target):
        policy_loss = action_distribution.log_prob(action_samples) * target
        policy_loss += action_distribution.entropy() * 1e-1
        policy_loss = tf.reduce_mean(policy_loss)

        # apply regularization
        reg_value = tfc.layers.apply_regularization(regularizer, self.trainable_variables)
        return policy_loss  + reg_value


class ValueEstimator(Base):
    def __init__(self):
        super(ValueEstimator, self).__init__()
        self.pose_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, None)
        ])

        self.obs_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, None)
        ])

        self.integrator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, None)
        ])

    def call(self, inputs, training=None, mask=None):
        poses = inputs[0]
        joints = inputs[1]
        pose_logits = self.pose_net(poses, training=training)
        joint_logits = self.obs_net(joints, training=training)
        state = tf.concat([pose_logits, joint_logits], axis=0)
        integrator_feed = tf.reduce_mean(state, axis=0, keepdims=True)
        return tf.squeeze(self.integrator(integrator_feed, training=training))

    def compute_loss(self, output, target):
        return tf.losses.mean_squared_error(target, output)
