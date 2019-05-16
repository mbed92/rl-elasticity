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

        self.joint_process = tf.keras.Sequential([
            tf.keras.layers.Dense(256, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(64, None)
        ])

        self.action_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(32, None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(num_controls, None)
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

    def call(self, inputs, training=None, mask=None):
        obs = inputs[0]
        rgb = inputs[1]
        rgb_logits = self.rgb_process(rgb, training=training)
        state_logits = self.joint_process(obs, training=training)
        integrator_feed = tf.add(rgb_logits, state_logits)

        # estimate mean actions
        mean_actions = self.action_estimator(integrator_feed, training=training)

        # estimate log of std deviations
        log_std_devs = self.log_std_devs_estimator(integrator_feed, training=training)

        return mean_actions, log_std_devs

