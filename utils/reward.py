from .kinematics import *


def get_target_pose(sim, base_name, target_name):
    base_xyz = sim.data.get_body_xpos(base_name)
    base_quat = sim.data.get_body_xquat(base_name)
    target_xyz = sim.data.get_body_xpos(target_name)
    target_quat = sim.data.get_body_xquat(target_name)

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

    if g02 > 0.7 and g06 > 0.9 and g10 > 0.4 and \
       g03 > 0.7 and g07 > 0.9 and g11 > 0.4 and \
       g04 > 0.7 and g08 > 0.9 and g12 > 0.4:
        return True
    return False


def get_reward(sim):

    # get the poses of targets in the robot's base coordinate system
    tool = get_target_pose(sim, 'base_link', 'gripperpalm')
    grip = get_target_pose(sim, 'base_link', 'CB17')
    body = get_target_pose(sim, 'base_link', 'fix')

    # reward from decreasing the distance between a gripper and a grip point
    grip_tool_dist = np.sum(np.abs(grip[0] - tool[0]))
    position_rew = 1 / grip_tool_dist if grip_tool_dist > 0.05 else 100

    # reward from stretching the object to the specified point
    grip_body_dist = np.sum(np.abs(grip[0] - body[0]))
    obj_reward = 1 / grip_body_dist if grip_body_dist > 0.05 else 100

    # reward from grasping the object (if it's close enough)
    grip_reward = 0
    if is_closed(sim) and grip_tool_dist < 0.05:
        grip_reward = 100

    return position_rew, obj_reward, grip_reward
