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
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()

    # make the policy network
    model = ContinuousAgent(num_controls=env.num_actions)
    optimizer, ckpt = setup_optimizer(args.restore_path, args.learning_rate, model)

    # run training
    for epoch in range(args.epochs):
        ep_rewards = []         # list for rewards accrued throughout ep
        ep_log_grad = []        # list of log-likelihood gradients
        batch_reward = []       # list of discounted and standardized sums rewards per epoch
        batch_means = []        # list of discounted and standardized means rewards per epoch
        total_gradient = []     # list of gradients multiplied by rewards per epochs
        keep_random = update_keep_random(args.keep_random, epoch, args.epochs)

        # domain randomization after each epoch
        env.randomize_environment()
        env.reset()

        # start trajectory
        cnt = 0
        while True:
            rgb, poses, joints = env.get_observations()
            cv2.imshow("trajectory", rgb[0])
            cv2.waitKey(1)

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as tape:
                logits = model([rgb, poses, joints], True)
                actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)
                actions = env.take_discrete_action(actions)
                env.step()

                ep_rew, distance_object = env.get_reward()
                ep_rew -= np.abs(0.008 * np.matmul(actions, np.transpose(actions)))
                ep_rewards.append(ep_rew)
                loss_value = tf.losses.mean_squared_error(ep_mean_act, actions)

            # compute and store gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            ep_log_grad = [tf.add(x, y) for x, y in zip(ep_log_grad, grads)] if len(ep_log_grad) != 0 else grads

            # compute grad log-likelihood for a current episode
            if distance_object > args.sim_max_dist or cnt > args.sim_max_length:
                if len(ep_rewards) > 5:
                    ep_rewards = standardize_rewards(ep_rewards)
                    ep_rewards = bound_to_nonzero(ep_rewards)
                    ep_rewards = discount_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = np.sum(ep_rewards), np.mean(ep_rewards)
                    batch_reward.append(ep_reward_sum)
                    batch_means.append(ep_reward_mean)
                    total_gradient = [tf.add(x, y) for x, y in zip(total_gradient, ep_log_grad)] if len(total_gradient) != 0 else ep_log_grad
                    print("Episode is done! Sum reward: {0}, mean reward: {1}, keep random ratio: {2}".format(ep_reward_sum, ep_reward_mean, keep_random))

                # reset episode-specific variables
                cnt = 0
                ep_rewards, ep_log_grad, weighted_grads = [], [], []
                if len(batch_reward) >= args.update_step:
                    break
                else:
                    env.reset()
            cnt += 1

        # take a single policy gradient update step
        num_episodes = len(batch_reward)
        rew_sum = sum(batch_reward)
        rew_mean = sum(batch_means) / num_episodes

        # get gradients and apply them to model's variables - gradient is computed as a mean from episodes
        total_gradient = [(a * (rew_sum - rew_mean)) / num_episodes for a in total_gradient]
        optimizer.apply_gradients(zip(total_gradient, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

        # update summary
        with tfc.summary.always_record_summaries():
            print('Epoch {0} finished! Training reward {1}'.format(epoch, rew_mean))
            tfc.summary.scalar('metric/reward', rew_mean, step=epoch)
            tfc.summary.scalar('metric/distance_object', distance_object, step=epoch)
            train_writer.flush()

        # save model
        if epoch % args.model_save_interval == 0:
            ckpt.save(os.path.join(args.save_path, 'ckpt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--model-save-interval', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--update-step', type=int, default=1)
    parser.add_argument('--sim-step', type=int, default=10)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=200)
    parser.add_argument('--sim-max-dist', type=float, default=0.15)
    parser.add_argument('--restore-path', type=str, default='./saved')
    parser.add_argument('--save-path', type=str, default='./saved')
    parser.add_argument('--logs-path', type=str, default='./log')
    parser.add_argument('--keep-random', type=float, default=0.7)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    os.makedirs(args.save_path, exist_ok=True)
    train(args)
