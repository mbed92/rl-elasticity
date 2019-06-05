from .interface import Env
from utils.kinematics import *
from constants import *

import mujoco_py
import numpy as np
import tensorflow as tf


x_range = (0.4, 0.6)
y_range = (-0.5, 0.0)
z_range = (0.5, 0.8)
EPS = 1e-8


class ManEnv(Env):
    def __init__(self, sim_start, sim_step, env_path, cam_id, img_width, img_height, base, tool, target):
        super().__init__(sim_start, sim_step)
        self.poses = []
        self.joints = []
        self.images = []
        self.cam_id = cam_id
        self.img_width = img_width
        self.img_height = img_height
        self.random_target = np.array([sum(x_range) / 2, sum(y_range) / 2, sum(z_range) / 2])
        self.link_base_name = base
        self.link_tool_name = tool
        self.link_trgt_name = target

        # setup environment and viewer
        scene = mujoco_py.load_model_from_path(env_path)
        self.env = mujoco_py.MjSim(scene)
        self.num_actions = self.env.data.ctrl.size
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.env, self.cam_id)

    # main methods
    def get_reward(self, actions):
        tool = self._get_target_pose(self.link_base_name, self.link_tool_name)
        target = self.random_target

        d2 = np.linalg.norm(target - tool[0])
        huber = d2 if d2 < 0.2 else np.square(d2)

        huber -= np.abs(0.005 * np.abs(np.matmul(actions, np.transpose(actions))))

        return huber, d2

    def step(self, num_steps=-1):
        if num_steps < 1:
            num_steps = self.sim_step
        try:
            for _ in range(num_steps):
                self.env.step()
        except mujoco_py.builder.MujocoException:
            self.reset()

    def reset(self):
        self.env.reset()

        for key, value in start_qpos.items():
            self.env.data.set_joint_qpos(key, value)

        if self.sim_start > 0:
            self.step(self.sim_start)

    def get_observations(self):
        self._get_camera_image()
        self._get_poses()
        self._get_joints()
        return self.images, self.poses, self.joints

    def take_continuous_action(self, means, std_devs, keep_prob):
        if np.random.uniform() < keep_prob:
            actions = tf.random_normal(tf.shape(means), mean=means, stddev=std_devs)
        else:
            actions = tf.random_uniform(tf.shape(means), means - 3 * std_devs, means + 3 * std_devs)
        for i in range(self.num_actions):
            self.env.data.ctrl[i] += actions.numpy()[0, i]
        return actions

    def take_discrete_action(self, actions):
        for i in range(self.num_actions):
            if actions[i] == 1:
                self.env.data.ctrl[i] = 0.25
            else:
                self.env.data.ctrl[i] = 0.0

    def randomize_environment(self):
        # self._randomize_rope_position()
        self._set_random_target()

    # specs
    @staticmethod
    def get_std_spec(args):
        return {
            "sim_start": args.sim_start,
            "sim_step": args.sim_step,
            "env_path": args.mujoco_model_path,
            "cam_id": args.sim_cam_id,
            "img_width": args.sim_cam_img_w,
            "img_height": args.sim_cam_img_h,
            "base": "base_link",
            "tool": "gripperpalm",
            "target": "CB8"
        }

    def _get_target_pose(self, base_name, target_name, offset=None):
        base_xyz = self.env.data.get_body_xpos(base_name)
        base_quat = self.env.data.get_body_xquat(base_name)
        target_xyz = self.env.data.get_body_xpos(target_name)
        target_quat = self.env.data.get_body_xquat(target_name)
        if offset is np.array and offset.size == target_xyz.size:
            for i in range(offset.size):
                target_xyz[i] += offset[i]

        translation = target_xyz - base_xyz
        rotation_quat = quaternion_multiply(target_quat, quaternion_inverse(base_quat))

        return translation, rotation_quat

    def _set_random_target(self):
        self.random_target[0] = np.random.uniform(x_range[0], x_range[1])
        self.random_target[1] = np.random.uniform(y_range[0], y_range[1])
        self.random_target[2] = np.random.uniform(z_range[0], z_range[1])

    def _randomize_rope_position(self):
        self.env.model.body_pos[1][1] = (np.random.uniform() - 0.5) / 10
        self.env.forward()

    def _get_camera_image(self):
        self.viewer.render(self.img_width, self.img_height, self.cam_id)
        rgb = np.asarray(self.viewer.read_pixels(self.img_width, self.img_height, depth=False)[::-1, :, :], dtype=np.float32)
        rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb / 255.
        self.images = np.float32(rgb[np.newaxis, :, :, :])

    def _get_poses(self):
        poses = list()

        p1 = self._get_target_pose(self.link_base_name, self.link_tool_name)[0]
        # p2 = self._get_target_pose(self.link_base_name, self.link_trgt_name)[0]
        # p2 = np.zeros_like(p1)
        p3 = self.random_target

        poses.append(p1)
        # poses.append(p2)
        poses.append(p3)
        poses = np.asarray(poses)
        self.poses = np.float32(poses[np.newaxis, :])

    def _get_joints(self):
        joints = list()
        joints.append(self.env.data.get_joint_qpos("shoulder_pan_joint"))
        joints.append(self.env.data.get_joint_qpos("shoulder_lift_joint"))
        joints.append(self.env.data.get_joint_qpos("elbow_joint"))
        joints.append(self.env.data.get_joint_qpos("wrist_1_joint"))
        joints.append(self.env.data.get_joint_qpos("wrist_2_joint"))
        joints.append(self.env.data.get_joint_qpos("wrist_3_joint"))
        joints.append(self.env.data.get_joint_qpos("gripperpalm_finger_1_joint"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_1_joint_1"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_1_joint_2"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_1_joint_3"))
        joints.append(self.env.data.get_joint_qpos("gripperpalm_finger_2_joint"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_2_joint_1"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_2_joint_2"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_2_joint_3"))
        joints.append(self.env.data.get_joint_qpos("gripperpalm_finger_middle_joint"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_middle_joint_1"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_middle_joint_2"))
        joints.append(self.env.data.get_joint_qpos("gripperfinger_middle_joint_3"))
        joints = np.asarray(joints)
        self.joints = np.float32(joints[np.newaxis, :])
