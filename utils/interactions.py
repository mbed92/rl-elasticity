import numpy as np
import mujoco_py
from .constants import start_qpos
from .reward import get_target_pose


EPS = 1e-8


def step(env, start_frame):
    try:
        for _ in range(start_frame):
            env.step()
    except mujoco_py.builder.MujocoException:
        step(env, start_frame)


def reset(env, start_frame):
    env.reset()
    env.data.set_joint_qpos("shoulder_pan_joint", start_qpos["shoulder_pan_joint"])
    env.data.set_joint_qpos("shoulder_lift_joint", start_qpos["shoulder_lift_joint"])
    env.data.set_joint_qpos("elbow_joint", start_qpos["elbow_joint"])
    env.data.set_joint_qpos("wrist_1_joint", start_qpos["wrist_1_joint"])
    env.data.set_joint_qpos("wrist_2_joint", start_qpos["wrist_2_joint"])
    env.data.set_joint_qpos("wrist_3_joint", start_qpos["wrist_3_joint"])

    if start_frame > 0:
        step(env, start_frame)


def get_camera_image(viewer, cam_id, width=320, height=240, normalize=True):
    viewer.render(width, height, cam_id)
    rgb = np.asarray(viewer.read_pixels(width, height, depth=False)[::-1, :, :], dtype=np.float32)
    if normalize:
        rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb / 255.
    return np.float32(rgb[np.newaxis, :, :, :])


def get_observations(sim):
    obs = list()
    poses = list()
    obs.append(sim.data.get_joint_qpos("shoulder_pan_joint"))
    obs.append(sim.data.get_joint_qpos("shoulder_lift_joint"))
    obs.append(sim.data.get_joint_qpos("elbow_joint"))
    obs.append(sim.data.get_joint_qpos("wrist_1_joint"))
    obs.append(sim.data.get_joint_qpos("wrist_2_joint"))
    obs.append(sim.data.get_joint_qpos("wrist_3_joint"))
    obs.append(sim.data.get_joint_qpos("gripperpalm_finger_1_joint"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_1_joint_1"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_1_joint_2"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_1_joint_3"))
    obs.append(sim.data.get_joint_qpos("gripperpalm_finger_2_joint"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_2_joint_1"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_2_joint_2"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_2_joint_3"))
    obs.append(sim.data.get_joint_qpos("gripperpalm_finger_middle_joint"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_middle_joint_1"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_middle_joint_2"))
    obs.append(sim.data.get_joint_qpos("gripperfinger_middle_joint_3"))
    poses.append(get_target_pose(sim, 'base_link', 'gripperpalm')[0])
    poses.append(get_target_pose(sim, 'base_link', 'CB17')[0])
    obs = np.asarray(obs)
    poses = np.asarray(poses)
    return np.float32(obs[np.newaxis, :]), np.float32(poses[np.newaxis, :])


def is_ep_done(reward):
    if reward < -1.0 or reward > 18:
        return True
    return False
