import numpy as np
import mujoco_py


EPS = 1e-8


def step(env, num):
    try:
        for _ in range(num):
            env.step()
    except mujoco_py.builder.MujocoException:
        pass


def get_camera_image(viewer, cam_id, width=320, height=240, normalize=True):
    viewer.render(width, height, cam_id)
    rgb = np.asarray(viewer.read_pixels(width, height, depth=False)[::-1, :, :], dtype=np.float32)
    if normalize:
        rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb / 255.
    return np.float32(rgb[np.newaxis, :, :, :])


def get_observations(sim):
    obs = list()
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
    obs = np.asarray(obs)
    return np.float32(obs[np.newaxis, :])


def is_ep_done(reward):
    # check last reward if in proper range
    if reward < 2.7 or reward > 299:
        return True
    return False
