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

        self.rgb_process = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (3, 3), 4, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, (3, 3), 4, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, None)
        ])

        self.pose_process = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, None)
        ])

        self.joints_process = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, None)
        ])

        self.RNN = tf.keras.layers.LSTMCell(128)

        self.action_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_controls, None)
        ])

        self.log_std_devs_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_controls, None)
        ])

        self.hidden_state = None

    def call(self, inputs, training=None, mask=None):
        # rgb = inputs[0]
        poses = inputs[0]
        joints = inputs[1]

        # rgb_logits = self.rgb_process(rgb, training=training)
        pos_logits = self.pose_process(poses, training=training)
        joi_logits = self.joints_process(poses, training=training)

        state = tf.concat([pos_logits, joi_logits], axis=0)
        integrator_feed = tf.reduce_mean(state, axis=0, keepdims=True)

        # add a flavour of a history
        self.hidden_state = self.RNN.get_initial_state(batch_size=tf.shape(integrator_feed)[0],
                                                       dtype=integrator_feed.dtype) if self.hidden_state is None else self.hidden_state
        logits, self.hidden_state = self.RNN(integrator_feed, states=self.hidden_state, training=training)
        # logits = self.RNN(integrator_feed)

        # estimate mean actions (-inf, inf)
        mean_actions = self.action_estimator(logits, training=training)

        # estimate log of std deviations (-inf, inf)
        log_std_devs = self.log_std_devs_estimator(logits, training=training)

        return mean_actions, log_std_devs

    def compute_loss(self, ep_std_dev, ep_variance, ep_mean_act, actions, regularier):
        policy_loss = tf.log((1 / ep_std_dev) + 1e-05) - (1 / ep_variance) * tf.losses.mean_squared_error(ep_mean_act, actions)
        policy_loss = tf.reduce_mean(policy_loss)

        # apply regularization
        reg_value = tfc.layers.apply_regularization(regularier, self.trainable_variables)
        return policy_loss + reg_value


class ValueEstimator(Base):
    def __init__(self):
        super(ValueEstimator, self).__init__()
        self.pose_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, None)
        ])

        self.obs_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, None)
        ])

        self.integrator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
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

    def compute_loss(self, reward_sums, baseline):
        return tf.losses.mean_squared_error(reward_sums, baseline)
