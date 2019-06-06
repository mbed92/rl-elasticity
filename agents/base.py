import tensorflow as tf


class ContinuousAgent(tf.keras.Model):

    def __init__(self, num_controls):
        super(ContinuousAgent, self).__init__()

        self.rgb_process = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), 2, 'same', activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(32, (5, 5), 4, 'same', activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3), 2, 'same', activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(128, (5, 5), 4, 'same', activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(256, (3, 3), 2, 'same', activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dense(32, None)
        ])

        self.pose_process = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, None)
        ])

        self.joint_process = tf.keras.Sequential([
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, None)
        ])

        self.RNN = tf.keras.layers.LSTMCell(32)

        self.action_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(32, tf.nn.relu),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_controls, None)
        ])

        self.log_std_devs_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(32, tf.nn.relu),
            tf.keras.layers.Dense(64, tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_controls, None)
        ])

        self.hidden_state = None

    def call(self, inputs, training=None, mask=None):
        rgb = inputs[0]
        poses = inputs[1]

        rgb_logits = self.rgb_process(rgb, training=training)
        pos_logits = self.pose_process(poses, training=training)

        state = tf.concat([rgb_logits, pos_logits], axis=0)
        integrator_feed = tf.reduce_sum(state, axis=0, keepdims=True)

        # add a flavour of a history
        self.hidden_state = self.RNN.get_initial_state(batch_size=tf.shape(integrator_feed)[0],
                                                       dtype=integrator_feed.dtype) if self.hidden_state is None else self.hidden_state
        logits, self.hidden_state = self.RNN(integrator_feed, states=self.hidden_state, training=training)

        # estimate mean actions
        mean_actions = self.action_estimator(logits, training=training)

        # estimate log of std deviations
        log_std_devs = self.log_std_devs_estimator(logits, training=training)

        return mean_actions, log_std_devs


class DiscreteAgent(ContinuousAgent):

    def call(self, inputs, training=None, mask=None):
        rgb = inputs[0]
        poses = inputs[1]
        joints = inputs[2]

        rgb_logits = self.rgb_process(rgb, training=training)
        pos_logits = self.pose_process(poses, training=training)
        state_logits = self.joint_process(joints, training=training)
        state = tf.concat([state_logits, rgb_logits, pos_logits], axis=0)
        integrator_feed = tf.reduce_mean(state, axis=0, keepdims=True)

        # add a flavour of a history
        self.hidden_state = self.RNN.get_initial_state(batch_size=tf.shape(integrator_feed)[0], dtype=integrator_feed.dtype) if self.hidden_state is None else self.hidden_state
        logits, self.hidden_state = self.RNN(integrator_feed, states=self.hidden_state, training=training)

        # estimate mean actions
        logits = self.action_estimator(logits, training=training)

        return logits
