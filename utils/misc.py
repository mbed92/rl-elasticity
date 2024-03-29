import os
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np


def setup_writer(log_path):
    os.makedirs(log_path, exist_ok=True)
    return tfc.summary.create_file_writer(log_path)


def setup_optimizer(restore_path, eta, model):
    optimizer = tf.train.AdamOptimizer(learning_rate=eta)
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


def is_done(current_distance, current_reward, t, args):
    if args.sim_max_dist < current_distance or current_distance < args.sim_min_dist or \
            current_reward > 9. or \
            t > args.sim_max_length:
        return True
    return False


def process_rewards(rewards):
    r = rewards
    r = bound_to_nonzero(r)
    r = reward_to_go(r)
    r = discount_rewards(r)
    r = standardize_rewards(r)
    return r
