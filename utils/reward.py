from scipy.special import huber

from .kinematics import *
from .constants import gripper_close


def get_target_pose(sim, base_name, target_name):
    base_xyz = sim.data.get_body_xpos(base_name)
    base_quat = sim.data.get_body_xquat(base_name)
    target_xyz = sim.data.get_body_xpos(target_name)
    target_quat = sim.data.get_body_xquat(target_name)
    if target_name == "gripperpalm":
        target_xyz[-1] += 0.2

    translation = target_xyz - base_xyz
    rotation_quat = quaternion_multiply(target_quat, quaternion_inverse(base_quat))

    return translation, rotation_quat


def discount_rewards(r, gamma=0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    r = np.asarray(r)
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def is_closed(sim):
    g02 = sim.data.get_joint_qpos("gripperfinger_1_joint_1")
    g03 = sim.data.get_joint_qpos("gripperfinger_1_joint_2")
    g04 = sim.data.get_joint_qpos("gripperfinger_1_joint_3")
    g06 = sim.data.get_joint_qpos("gripperfinger_2_joint_1")
    g07 = sim.data.get_joint_qpos("gripperfinger_2_joint_2")
    g08 = sim.data.get_joint_qpos("gripperfinger_2_joint_3")
    g10 = sim.data.get_joint_qpos("gripperfinger_middle_joint_1")
    g11 = sim.data.get_joint_qpos("gripperfinger_middle_joint_2")
    g12 = sim.data.get_joint_qpos("gripperfinger_middle_joint_3")

    if g02 > gripper_close["gripperfinger_1_joint_1"] and g06 > gripper_close["gripperfinger_2_joint_1"] and g10 > gripper_close["gripperfinger_middle_joint_1"] and \
       g03 > gripper_close["gripperfinger_1_joint_2"] and g07 > gripper_close["gripperfinger_2_joint_2"] and g11 > gripper_close["gripperfinger_middle_joint_2"] and \
       g04 > gripper_close["gripperfinger_1_joint_3"] and g08 > gripper_close["gripperfinger_2_joint_3"] and g12 > gripper_close["gripperfinger_middle_joint_3"]:
        return True
    return False


def get_reward(sim, u):

    # get the poses of targets in the robot's base coordinate system
    tool = get_target_pose(sim, 'base_link', 'gripperpalm')
    grip = get_target_pose(sim, 'base_link', 'CB17')
    # body = get_target_pose(sim, 'base_link', 'fix')

    # reward from decreasing the distance between a gripper and a grip point - reach reward
    d = np.linalg.norm(grip[0] - tool[0])
    grip_tool_dist = d if d < 0.3 else np.square(d)  # Huber loss
    gamma_1, gamma_2 = 0.9, 0.01
    position_rew = -gamma_1 * grip_tool_dist - gamma_2 * np.matmul(u, np.transpose(u))

    # add a sparse rewards
    position_rew = (position_rew + 1) if d < 0.6 else position_rew
    position_rew = (position_rew + 2) if d < 0.5 else position_rew
    position_rew = (position_rew + 3) if d < 0.4 else position_rew
    position_rew = (position_rew + 5) if d < 0.3 else position_rew
    position_rew = (position_rew + 8) if d < 0.2 else position_rew

    # reward from stretching the object to the specified point
    # grip_body_dist = np.sum(np.abs(grip[0] - body[0]))
    # obj_reward = 1 / grip_body_dist if grip_body_dist > 0.1 else 100
    # position_rew *= 0.6
    # if grip_tool_dist < 0.2 and grip_body_dist < 0.2:
    #     position_rew *= 1.3
    #     obj_reward *= 1.3
    # reward from grasping the object (if it's close enough)
    # grip_reward = 0
    # if is_closed(sim) and grip_tool_dist < 0.2:
    #     grip_reward = 100
    # print(d, position_rew)

    return position_rew  # , obj_reward, grip_reward


def standarize_rewards(rewards: list):
    rewards = np.asarray(rewards)

    # standarize
    m, s = np.mean(rewards), np.sqrt(np.var(rewards))
    ret = (rewards - m) / (s + 1e-06)

    # set to range from abs(minimal value) - 2* because first action has almost always the lowest reward and
    # would be set to 0 (environment specific)
    # ret -= 2*np.min(ret)
    return ret.tolist()
