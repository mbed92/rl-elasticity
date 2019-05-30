import os
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from agents import Core
from utils import *

tf.enable_eager_execution()
tf.executing_eagerly()


def train(args):
    env, viewer = setup_environment(args.mujoco_model_path)
    train_writer = setup_writer(args.logs_path)
    train_writer.set_as_default()

    # make core of policy network
    num_inputs = env.data.ctrl.size
    model = Core(num_controls=num_inputs)

    # create optimizer for the apply_gradients() and load weights
    optimizer, ckpt = setup_optimizer(args.restore_path, args.learning_rate, model)

    # run training
    for epoch in range(args.epochs):
        ep_rewards = []         # list for rewards accrued throughout ep
        ep_log_grad = []        # list of log-likelihood gradients
        batch_reward = []       # list of discounted and standarized sums rewards per epoch
        batch_means = []        # list of discounted and standarized means rewards per epoch
        total_gradient = []     # list of gradients multiplied by rewards per epochs

        # randomize final point in every epoch
        set_random_target()

        # start trajectory
        cnt = 0
        randomize_target(env)
        reset(env, args.sim_start)
        keep_random = update_keep_random(args.keep_random, epoch, args.epochs)
        while True:
            obs, pos = get_observations(env)    # TODO: obs and rgb move to one fcn
            rgb = get_camera_image(viewer, cam_id=0)
            cv2.imshow("trajectory", rgb[0])
            cv2.waitKey(1)

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as tape:
                ep_mean_act, ep_log_dev = model([obs, rgb, pos], True)
                ep_stddev = tf.exp(ep_log_dev)

                # apply actions # TODO: move to one function taking_actions()
                if np.random.uniform() < keep_random:
                    actions = tf.random_normal(tf.shape(ep_mean_act), mean=ep_mean_act, stddev=ep_stddev)
                else:
                    actions = tf.random_uniform(tf.shape(ep_mean_act), ep_mean_act - 3 * ep_stddev, ep_mean_act + 3 * ep_stddev)
                for i in range(num_inputs):
                    env.data.ctrl[i] += actions.numpy()[0, i]

                # act, compute reward and loss
                step(env, args.sim_step)

                # ep_rew, distance = get_distance_reward(env, actions.numpy())
                ep_rew, distance_object, distance_target = get_sparse_reward(env)
                ep_rewards.append(ep_rew)
                loss_value = tf.losses.mean_squared_error(ep_mean_act, actions)

            # compute and store gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            ep_log_grad = [tf.add(x, y) for x, y in zip(ep_log_grad, grads)] if len(ep_log_grad) != 0 else grads

            # compute grad log-likelihood for a current episode
            if distance_object > 0.8 or cnt > 150:
                if len(ep_rewards) > 5:  # do not accept one-element lists of rewards or trash moves
                    ep_rewards = discount_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = np.sum(ep_rewards), np.mean(ep_rewards)
                    batch_reward.append(ep_reward_sum)
                    batch_means.append(ep_reward_mean)
                    total_gradient = [tf.add(x, y) for x, y in zip(total_gradient, ep_log_grad)] if len(total_gradient) != 0 else ep_log_grad
                    print("Episode is done! Sum reward: {0}, mean reward: {1}, keep random ratio: {2}".format(ep_reward_sum, ep_reward_mean, keep_random))

                # reset episode-specific variables
                ep_rewards, ep_log_grad, weighted_grads = [], [], []
                cnt = 0
                if len(batch_reward) >= args.update_step:
                    break
                else:
                    reset(env, args.sim_start)
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
            tfc.summary.image('scene/camera_img', rgb, max_images=1, step=epoch)
            tfc.summary.scalar('metric/reward', rew_mean, step=epoch)
            tfc.summary.scalar('metric/distance_object', distance_object, step=epoch)
            tfc.summary.scalar('metric/distance_target', distance_target, step=epoch)
            print('Epoch {0} finished! Training reward {1}'.format(epoch, rew_mean))
            train_writer.flush()

        # save model
        if epoch % 100 == 0:
            ckpt.save(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--update-step', type=int, default=1)
    parser.add_argument('--sim-step', type=int, default=10)
    parser.add_argument('--sim-start', type=int, default=1000)
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--save-path', type=str, default='./saved')
    parser.add_argument('--logs-path', type=str, default='./log')
    parser.add_argument('--keep-random', type=float, default=0.7)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()

    train(args)
