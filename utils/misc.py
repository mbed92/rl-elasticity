import os
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np


def setup_writer(log_path):
    os.makedirs(log_path, exist_ok=True)
    return tfc.summary.create_file_writer(log_path)


def setup_optimizer(restore_path, lr, model):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())

    if restore_path is not "":
        ckpt.restore(tf.train.latest_checkpoint(restore_path))
        print("Weight successfully restored")
    else:
        print("No model loaded. Training from the beginning.")

    return optimizer, ckpt


def update_keep_random(initial_keep_random, epoch, epochs):
    val = initial_keep_random + ((epoch / epochs) * (1 - initial_keep_random))
    return val if 0.0 < val < 1.0 else 1.0


def standardize_rewards(rewards):
    rewards = np.asarray(rewards)
    m, s = np.mean(rewards), np.sqrt(np.var(rewards))
    return (rewards - m) / (s + 1e-06)


def bound_to_nonzero(rewards):
    lowest = np.abs(np.min(rewards))
    rewards += (2 * lowest)
    return rewards


def discount_rewards(r, gamma=0.98):
    discounted_r = np.zeros_like(r)
    r = np.asarray(r)
    for t in range(0, r.size):
        discounted_r[t] = np.power(gamma, r.size - t - 1) + r[t] if r[t] > 0 else r[t]
    return discounted_r
