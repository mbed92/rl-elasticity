import tensorflow as tf


class Core(tf.keras.Model):

    def __init__(self, num_controls):
        super(Core, self).__init__()

        self.rgb_process = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(64, None)
        ])

        self.pose_process = tf.keras.Sequential([
            tf.keras.layers.Dense(256, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, None)
        ])

        self.joint_process = tf.keras.Sequential([
            tf.keras.layers.Dense(256, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(64, None)
        ])

        self.RNN = tf.keras.layers.LSTMCell(64)

        self.action_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(32, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(num_controls, tf.nn.tanh)
        ])

        self.log_std_devs_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(32, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(num_controls, None)
        ])

        self.hidden_state = None

    def call(self, inputs, training=None, mask=None):
        obs = inputs[0]
        rgb = inputs[1]
        pos = inputs[2]

        pos_logits = self.pose_process(pos, training=training)
        rgb_logits = self.rgb_process(rgb, training=training)
        state_logits = self.joint_process(obs, training=training)
        integrator_feed = pos_logits + rgb_logits + state_logits

        # add a flavour of a history
        self.hidden_state = self.RNN.get_initial_state(batch_size=tf.shape(integrator_feed)[0], dtype=integrator_feed.dtype) if self.hidden_state is None else self.hidden_state
        logits, self.hidden_state = self.RNN(integrator_feed, states=self.hidden_state, training=training)

        # estimate mean actions
        mean_actions = self.action_estimator(logits, training=training)

        # estimate log of std deviations
        log_std_devs = self.log_std_devs_estimator(logits, training=training)

        return mean_actions, log_std_devs

