import os
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing
import sklearn
import time


def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


def setup_writer(log_path):
    os.makedirs(log_path, exist_ok=True)
    return tfc.summary.create_file_writer(log_path)


def setup_optimizer(restore_path, eta, model: tf.keras.Model, weights_file='policy_network.hdf5'):
    optimizer = tf.train.AdamOptimizer(learning_rate=eta)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())

    if restore_path is not "":
        latest = tf.train.latest_checkpoint(restore_path)
        ckpt.restore(latest)
        model.load_weights(os.path.join(restore_path, weights_file))
        print("Weight successfully restored")
    else:
        print("No model loaded. Training from the beginning.")

    return optimizer, ckpt


def update_keep_random(initial_keep_random, epoch, epochs):
    if initial_keep_random == 1.0:
        return 1.0
    val = initial_keep_random + ((epoch / epochs) * (1 - initial_keep_random))
    return val if 0.0 < val < 1.0 else 1.0


def standardize_rewards(rewards: list):
    m, s = np.mean(rewards), np.std(rewards)
    return [(rew - m) / (s + 1e-05) for rew in rewards]


def bound_to_nonzero(rewards: list):
    lowest = abs(min(rewards))
    rewards += (2 * lowest)
    return rewards


def discount_rewards(rewards: list, gamma=0.98):
    discounted_r = [0] * len(rewards)
    for t in range(len(rewards)):
        discounted_r[t] = gamma ** (len(rewards) - t - 1) * rewards[t] if rewards[t] > 0 else rewards[t]
    return discounted_r


def reward_to_go(rewards: list):
    n = len(rewards)
    rtgs = [0] * len(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def process_rewards(rewards):
    r = rewards
    # r = bound_to_nonzero(r)
    r = discount_rewards(r, gamma=1.02)
    r = reward_to_go(r)
    # r = standardize_rewards(r)
    return r


def get_fitter(env):
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    def _get_sample():
        poses, joints = env.get_observations()
        a = np.concatenate([np.reshape(poses, newshape=[-1]), np.reshape(joints, newshape=[-1])], 0)
        return a
    observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to convert a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    ])
    featurizer.fit(scaler.transform(observation_examples))
    return scaler, featurizer

def process_state(state, scaler, featurizer):
    # if len(state) == 2:
    #     state = np.concatenate([np.reshape(state[0], newshape=[-1]), np.reshape(state[1], newshape=[-1])], 0)
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0][np.newaxis, :]