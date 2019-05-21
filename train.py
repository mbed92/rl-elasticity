import os
import mujoco_py
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from agents import Core
from utils import *
import cv2


tf.enable_eager_execution()
tf.executing_eagerly()


def train(epochs=2000, num_ep_per_batch=1, lr=1e-04, step_size=10, start_frame=1000):
    # make environment, check spaces, get obs / act dims
    path = os.path.join('.', 'models', 'ur5', 'UR5gripper.xml')
    scene = mujoco_py.load_model_from_path(path)
    env = mujoco_py.MjSim(scene)
    viewer = mujoco_py.MjRenderContextOffscreen(env, 0)

    # setup writer
    train_log_path = os.path.join('.', 'logs')
    os.makedirs(train_log_path, exist_ok=True)
    train_writer = tfc.summary.create_file_writer(train_log_path)

    # make core of policy network
    model = Core(num_controls=env.data.ctrl.size)

    # create optimizer for the apply_gradients()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    checkpoint_prefix = os.path.join('.', 'saved', 'ckpt')
    os.makedirs(checkpoint_prefix, exist_ok=True)

    # run training
    keep_random = 0.8
    for epoch in range(epochs):
        train_writer.set_as_default()
        train_reward = tfc.eager.metrics.Mean('reward')

        # collect experience by acting in the environment with current policy
        ep_rewards = []          # list for rewards accrued throughout ep
        ep_log_grad = []         # list of log-likelihood gradients
        batch_reward = []
        batch_means = []
        total_gradient = []

        # start learning from the start_frame
        cnt = 0
        reset(env, start_frame)
        while True:
            obs, pos = get_observations(env)
            rgb = get_camera_image(viewer, cam_id=0)

            cv2.imshow("aaa", rgb[0])
            cv2.waitKey(1)

            # take action in the environment under the current policy
            with tf.GradientTape(persistent=True) as tape:
                ep_mean_act, ep_log_dev = model([obs, rgb, pos], True)
                ep_stddev = tf.exp(ep_log_dev)

                # apply actions
                if np.random.rand() > keep_random:
                    actions = tf.random_normal(tf.shape(ep_mean_act), mean=ep_mean_act, stddev=ep_stddev)
                else:
                    actions = tf.random_uniform(tf.shape(ep_mean_act), ep_mean_act - 3 * ep_stddev, ep_mean_act + 3 * ep_stddev)
                keep_random += (epoch/epochs)

                for i in range(len(env.data.ctrl)):
                    env.data.ctrl[i] += actions.numpy()[0, i]

                # speed up simulation
                step(env, step_size)

                # compute reward and loss
                ep_rew = sum(get_reward(env, actions.numpy()))
                # print(ep_rew)
                ep_rewards.append(ep_rew)
                loss_value = tf.losses.mean_squared_error(ep_mean_act, actions)

            # compute and store gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            ep_log_grad = [tf.add(x, y) for x, y in zip(ep_log_grad, grads)] if len(ep_log_grad) != 0 else grads

            # compute grad log-likelihood for a current episode
            if is_ep_done(ep_rew):
                if len(ep_rewards) > 10:  # do not accept one-element lists of rewards or trash moves
                    ep_rewards = standarize_rewards(ep_rewards)
                    ep_rewards -= np.min(ep_rewards)    # make rewards always > 0
                    ep_rewards = discount_rewards(ep_rewards)
                    ep_reward_sum, ep_reward_mean = sum(ep_rewards), np.mean(np.asarray(ep_rewards))
                    print("Episode is done! Sum reward: {0}, mean reward: {1}".format(ep_reward_sum, ep_reward_mean))

                    # normalize rewards and apply them to gradients
                    batch_reward.append(ep_reward_sum)
                    batch_means.append(ep_reward_mean)
                    total_gradient = [tf.add(x, y) for x, y in zip(total_gradient, ep_log_grad)] if len(total_gradient) != 0 else ep_log_grad

                # reset episode-specific variables
                ep_rewards, ep_log_grad, weighted_grads = [], [], []

                # end experience loop if we have enough of it
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
        total_gradient = [(a / num_episodes) * (rew_sum - rew_mean) for a in total_gradient]
        optimizer.apply_gradients(zip(total_gradient, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

        # update summary
        train_reward(rew_mean)
        print('Epoch {0} finished! Training reward {1}'.format(epoch, train_reward.result()))
        with tfc.summary.always_record_summaries():
            tfc.summary.image('scene/camera_img', rgb, max_images=1, step=epoch)
            tfc.summary.scalar('metric/reward', rew_mean, step=epoch)
            train_writer.flush()

        # save model
        if epoch % 60 == 0:
            ckpt.save(checkpoint_prefix)


if __name__ == '__main__':
    train()
