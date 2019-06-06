from argparse import ArgumentParser

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from agents import ContinuousAgent
from environment import ManEnv
from utils import *

tf.enable_eager_execution()
tf.executing_eagerly()


def train(args):
    global total_gradient

    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()

    # make the policy network
    model = ContinuousAgent(num_controls=env.num_actions)
    optimizer, ckpt = setup_optimizer(args.restore_path, args.learning_rate, model)
    l2_reg = tf.keras.regularizers.l2(1e-4)

    # run training
    for n in range(args.epochs):
        batch_rewards = []
        ep_rewards = []       # list for rewards accrued throughout ep
        ep_log_grad = []      # list of log-likelihood gradients
        total_gradient = []   # list of gradients multiplied by rewards per epochs
        keep_random = update_keep_random(args.keep_random, n, args.epochs)

        # domain randomization after each epoch
        env.randomize_environment()
        env.reset()

        # start trajectory
        t, trajs = 0, 0
        while True:
            rgb, poses, _ = env.get_observations()
            cv2.imshow("trajectory", rgb[0])
            cv2.waitKey(1)

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as tape:
                ep_mean_act, ep_log_dev = model([rgb, poses], True)
                ep_std_dev, ep_variance = tf.exp(ep_log_dev), tf.square(tf.exp(ep_log_dev))
                actions = env.take_continuous_action(ep_mean_act, ep_std_dev, keep_random)

                env.step()
                ep_rew, distance = env.get_reward(actions)
                ep_rewards.append(ep_rew)

                # optimize a mean and a std_dev
                loss_value = tf.log(1 / ep_std_dev) - (1 / ep_variance) * tf.losses.mean_squared_error(ep_mean_act, actions)
                loss_value = tf.reduce_mean(loss_value)

                # optimize only a mean
                # loss_value = -1.0 * tf.losses.mean_squared_error(ep_mean_act, actions)

                reg_value = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                loss = loss_value + reg_value

            # compute and store gradients
            grads = tape.gradient(loss, model.trainable_variables)
            ep_log_grad = [tf.add(x, y) for x, y in zip(ep_log_grad, grads)] if len(ep_log_grad) != 0 else grads
            t += 1

            # compute grad log-likelihood for a current episode
            if distance > args.sim_max_dist or distance < 0.05 or t > args.sim_max_length or ep_rew > 50:
                if len(ep_rewards) > 5:
                    ep_rewards = standardize_rewards(ep_rewards)
                    ep_rewards = bound_to_nonzero(ep_rewards)
                    ep_rewards = discount_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = sum(ep_rewards), mean(ep_rewards)
                    batch_rewards.append(ep_reward_mean)

                    # compute gradient and multiply it times "reward to go" minus baseline
                    total_gradient = [tf.add(log, prev_log) for log, prev_log in zip(total_gradient, ep_log_grad)] if len(total_gradient) != 0 else ep_log_grad
                    total_gradient = [tf.multiply(log, ep_reward_sum - ep_reward_mean) for log in total_gradient]
                    print("Episode is done! Sum reward: {0}, mean reward: {1}, keep random ratio: {2}".format(ep_reward_sum, ep_reward_mean, keep_random))
                    trajs += 1

                    if trajs >= args.update_step:
                        print("Apply gradients!")
                        break

                # reset episode-specific variables
                ep_rewards, ep_log_grad = [], []
                env.randomize_environment()
                env.reset()
                t = 0

        # get gradients and apply them to model's variables - gradient is computed as a mean from episodes
        total_gradient = [tf.div(grad, trajs) for grad in total_gradient] if trajs > 1 else total_gradient
        optimizer.apply_gradients(zip(total_gradient, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

        # update summary
        with tfc.summary.always_record_summaries():
            print('Epoch {0} finished!'.format(n))
            tfc.summary.scalar('metric/distance', distance, step=n)
            tfc.summary.scalar('metric/mean_reward', np.mean(batch_rewards), step=n)
            train_writer.flush()

        # save model
        if n % args.model_save_interval == 0:
            ckpt.save(os.path.join(args.save_path, 'ckpt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--model-save-interval', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--update-step', type=int, default=2)
    parser.add_argument('--sim-step', type=int, default=10)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=150)
    parser.add_argument('--sim-max-dist', type=float, default=1.5)
    parser.add_argument('--restore-path', type=str, default='')
    parser.add_argument('--save-path', type=str, default='./saved')
    parser.add_argument('--logs-path', type=str, default='./log/1')
    parser.add_argument('--keep-random', type=float, default=0.8548)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.save_path, exist_ok=True)
    train(args)
