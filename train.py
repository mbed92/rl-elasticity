import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from agents import Core
from utils import *

tf.enable_eager_execution()
tf.executing_eagerly()


def train(epochs=2000, num_ep_per_batch=1, lr=1e-04, step_size=10, start_frame=1000, restore=False):
    env, viewer = setup_environment()
    train_writer = setup_writer()
    train_writer.set_as_default()
    train_reward = tfc.eager.metrics.Mean('reward')
    train_distance = tfc.eager.metrics.Mean('distance')

    # make core of policy network
    num_inputs = env.data.ctrl.size
    model = Core(num_controls=num_inputs)

    # create optimizer for the apply_gradients() and load weights
    optimizer, ckpt = setup_optimizer(restore, lr, model)

    # run training
    for epoch in range(epochs):
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
        reset(env, start_frame)
        keep_random = initial_keep_random + ((epoch / epochs) / (1 / initial_keep_random))
        ep_rew = [0]
        while True:
            obs, pos = get_observations(env)
            rgb = get_camera_image(viewer, cam_id=0)
            cv2.imshow("trajectory", rgb[0])
            cv2.waitKey(1)

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as tape:
                ep_mean_act, ep_log_dev = model([obs, rgb, pos], True)
                ep_stddev = tf.exp(ep_log_dev)

                # apply actions
                if np.random.uniform() < keep_random:
                    actions = tf.random_normal(tf.shape(ep_mean_act), mean=ep_mean_act, stddev=ep_stddev)
                else:
                    actions = tf.random_uniform(tf.shape(ep_mean_act), ep_mean_act - 3 * ep_stddev, ep_mean_act + 3 * ep_stddev)
                for i in range(num_inputs):
                    env.data.ctrl[i] += actions.numpy()[0, i]

                # act, compute reward and loss
                step(env, step_size)

                # ep_rew, distance = get_distance_reward(env, actions.numpy())
                ep_rew, distance = get_sparse_reward(env, ep_rew)

                ep_rewards.append(sum(ep_rew))
                loss_value = tf.losses.mean_squared_error(ep_mean_act, actions)

            # compute and store gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            ep_log_grad = [tf.add(x, y) for x, y in zip(ep_log_grad, grads)] if len(ep_log_grad) != 0 else grads

            # compute grad log-likelihood for a current episode
            if distance > 0.8 or cnt > 400:
                if len(ep_rewards) > 5:  # do not accept one-element lists of rewards or trash moves
                    ep_reward_sum, ep_reward_mean = standarize_rewards(ep_rewards)
                    batch_reward.append(ep_reward_sum)
                    batch_means.append(ep_reward_mean)
                    total_gradient = [tf.add(x, y) for x, y in zip(total_gradient, ep_log_grad)] if len(total_gradient) != 0 else ep_log_grad
                    print("Episode is done! Sum reward: {0}, mean reward: {1}, keep random ratio: {2}".format(ep_reward_sum, ep_reward_mean, keep_random))

                # reset episode-specific variables
                ep_rewards, ep_log_grad, weighted_grads = [], [], []
                cnt = 0
                if len(batch_reward) >= num_ep_per_batch:
                    break
                else:
                    reset(env, start_frame)
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
            train_reward(rew_mean)
            train_distance(distance)
            tfc.summary.image('scene/camera_img', rgb, max_images=1, step=epoch)
            tfc.summary.scalar('metric/reward', train_reward.result(), step=epoch)
            tfc.summary.scalar('metric/distance', train_distance.result(), step=epoch)
            print('Epoch {0} finished! Training reward {1}'.format(epoch, train_reward.result()))
            train_writer.flush()

        # save model
        if epoch % 100 == 0:
            ckpt.save(os.path.join(*model_nn))


if __name__ == '__main__':
    train()
