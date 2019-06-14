from argparse import ArgumentParser

import cv2
import tensorflow as tf

from agents import PolicyNetwork
from environment import ManEnv

tf.enable_eager_execution()
tf.executing_eagerly()


def test(args):
    env_spec = ManEnv.get_std_spec(args)
    env = ManEnv(**env_spec)
    model = PolicyNetwork(num_controls=env.num_actions)

    # randomize_target(env)
    env.reset()
    for i in range(2000):
        rgb, poses, joints = env.get_observations()
        cv2.imshow("trajectory", rgb[0])
        cv2.waitKey(1)

        ep_mean_act, ep_log_dev = model([rgb, poses, joints], True)
        ep_stddev = tf.exp(ep_log_dev)
        env.take_continuous_action(ep_mean_act, ep_stddev, 1.0)
        env.step()

        if i % 100 == 0:
            env.randomize_environment()
            env.reset()


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
    parser.add_argument('--sim-max-dist', type=float, default=0.8)
    parser.add_argument('--restore-path', type=str, default="")
    parser.add_argument('--save-path', type=str, default='./saved')
    parser.add_argument('--logs-path', type=str, default='./log')
    parser.add_argument('--keep-random', type=float, default=0.7)
    parser.add_argument('--mujoco-model-path', type=str, default='./models/ur5/UR5gripper.xml')
    args, _ = parser.parse_known_args()
    test(args)
