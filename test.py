import os

import cv2
import mujoco_py
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from agents import Core
from utils import *

tf.enable_eager_execution()
tf.executing_eagerly()


def test(step_size=5, start_frame=1000):
    # make the environment
    path = os.path.join('.', 'models', 'ur5', 'UR5gripper.xml')
    scene = mujoco_py.load_model_from_path(path)
    env = mujoco_py.MjSim(scene)
    viewer = mujoco_py.MjRenderContextOffscreen(env, 0)

    # make core of policy network
    model = Core(num_controls=env.data.ctrl.size)

    # load weights
    dir = os.path.join('.', 'saved')
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(dir))

    randomize_target(env)
    reset(env, start_frame)
    for i in range(1000):
        obs, pos = get_observations(env)
        rgb = get_camera_image(viewer, cam_id=0)

        cv2.imshow("trajectory", rgb[0])
        cv2.waitKey(1)

        ep_mean_act, ep_log_dev = model([obs, rgb, pos], False)
        ep_stddev = tf.exp(ep_log_dev)
        actions = tf.random_normal(tf.shape(ep_mean_act), mean=ep_mean_act, stddev=ep_stddev)

        for k in range(len(env.data.ctrl)):
            env.data.ctrl[k] = actions.numpy()[0, k]

        # speed up simulation
        step(env, step_size)

        if i % 200 == 0:
            randomize_target(env)
            reset(env, start_frame)


if __name__ == '__main__':
    test()