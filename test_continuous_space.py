from argparse import ArgumentParser

import cv2
import tensorflow as tf
import mujoco_py
import numpy as np

from agents import PolicyNetwork
from environment import ManEnv

tf.enable_eager_execution()
tf.executing_eagerly()


def test(args):
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    policy = PolicyNetwork(num_controls=env.num_actions)

    ckpt_policy = tf.train.Checkpoint(model=policy)
    ckpt_policy.restore(tf.train.latest_checkpoint(args.policy_restore_path))

    viewer = mujoco_py.MjRenderContextOffscreen(env.env, 0)
    env.reset()
    for i in range(2000):
        poses, joints = env.get_observations()
        actions_distribution = policy([poses, joints], True)
        actions = env.take_continuous_action(actions_distribution, 1.0)
        ep_rew, distance = env.get_reward(actions)
        env.step()

        rgb = np.asarray(viewer.read_pixels(640, 480, depth=False)[::-1, :, :], dtype=np.float32)
        rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb / 255.
        rgb = np.float32(rgb[np.newaxis, :, :, :])
        cv2.imshow("aaa", rgb[0])
        cv2.waitKey(1)

        if i % 100 == 0 or distance < 0.05:
            env.randomize_environment()
            print(distance)
            env.reset()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--model-save-interval', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--update-step', type=int, default=1)
    parser.add_argument('--sim-step', type=int, default=5)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-cam-id', type=int, default=0)
    parser.add_argument('--sim-cam-img-w', type=int, default=640)
    parser.add_argument('--sim-cam-img-h', type=int, default=480)
    parser.add_argument('--sim-max-length', type=int, default=100)
    parser.add_argument('--sim-max-dist', type=float, default=1.5)
    parser.add_argument('--sim-min-dist', type=float, default=0.05)
    parser.add_argument('--policy-restore-path', type=str, default='./saved/policy')
    parser.add_argument('--value-restore-path', type=str, default='./saved/value')
    parser.add_argument('--policy-save-path', type=str, default='./saved/policy')
    parser.add_argument('--value-save-path', type=str, default='./saved/value')
    parser.add_argument('--logs-path', type=str, default='./log/1')
    parser.add_argument('--keep-random', type=float, default=0.5)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()
    test(args)
