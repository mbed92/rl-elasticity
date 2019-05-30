import mujoco_py
from .constants import *
from .kinematics import *


random_target = np.array([sum(x_range) / 2, sum(y_range) / 2, sum(z_range) / 2])
EPS = 1e-8


def get_target_pose(sim, base_name, target_name):
    base_xyz = sim.data.get_body_xpos(base_name)
    base_quat = sim.data.get_body_xquat(base_name)
    target_xyz = sim.data.get_body_xpos(target_name)
    target_quat = sim.data.get_body_xquat(target_name)
    if target_name == "gripperpalm":
        target_xyz[1] -= 0.06

    translation = target_xyz - base_xyz
    rotation_quat = quaternion_multiply(target_quat, quaternion_inverse(base_quat))

    return translation, rotation_quat


def get_random_target():
    global random_target
    return random_target


def set_random_target():
    global random_target
    random_target[0] = np.random.uniform(x_range[0], x_range[1])
    random_target[1] = np.random.uniform(y_range[0], y_range[1])
    random_target[2] = np.random.uniform(z_range[0], z_range[1])


def randomize_target(env):
    env.model.body_pos[1][1] = (2 * np.random.rand() - 1) / 10.0
    env.forward()
    return env


def step(env, start_frame):
    try:
        for _ in range(start_frame):
            env.step()
    except mujoco_py.builder.MujocoException:
        reset(env, 1000)


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


def get_camera_image(viewer, cam_id, width=640, height=420, normalize=True):
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
    poses.append(get_target_pose(sim, 'base_link', 'CB13')[0])
    poses.append(get_random_target())
    obs = np.asarray(obs)
    poses = np.asarray(poses)
    return np.float32(obs[np.newaxis, :]), np.float32(poses[np.newaxis, :])
