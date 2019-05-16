from .kinematics import *


def get_tool_pose(sim):
    base_xyz = sim.data.get_body_xpos('base_link')
    base_quat = sim.data.get_body_xquat('base_link')
    tool_xyz = sim.data.get_body_xpos('gripperpalm')
    tool_quat = sim.data.get_body_xquat('gripperpalm')

    translation = tool_xyz - base_xyz
    rotation_quat = quaternion_multiply(tool_quat, quaternion_inverse(base_quat))

    return translation, rotation_quat


def get_target_pose(sim):
    base_xyz = sim.data.get_body_xpos('base_link')
    base_quat = sim.data.get_body_xquat('base_link')
    target_xyz = sim.data.get_body_xpos('CB19')
    target_quat = sim.data.get_body_xquat('CB19')

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


def get_reward(sim, tool, target):

    # reward from position of tool
    pos_dist = np.sum(np.abs(target[0] - tool[0]))

    # reward from position of object
    obj_dist = np.sum(np.abs(target[0] - [0.25, 0, 0.875]))
    obj_reward = 1 / obj_dist if obj_dist > 0.05 else 100
    position_rew = 1 / pos_dist if pos_dist > 0.05 else 100

    # reward from grasping the object
    grip_reward = 0
    if is_closed(sim) and pos_dist < 0.1:
        grip_reward = 100

    return position_rew, grip_reward, obj_reward
