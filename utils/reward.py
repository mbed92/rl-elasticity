from .constants import gripper_close
from .interactions import *


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


def get_sparse_reward(sim, reward):
    tool = get_target_pose(sim, 'base_link', 'gripperpalm')
    grip = get_target_pose(sim, 'base_link', 'CB17')
    body = get_random_target()
    reward -= 1

    d = np.linalg.norm(grip[0] - tool[0])
    if d < 0.25:
        reward += 100
    if d < 0.05:
        reward += 1000

    d2 = np.linalg.norm(body - grip[0])
    if d2 < 0.2:
        reward += 100
    if d2 < 0.05:
        reward += 1000

    return reward


def get_distance_reward(sim, u):

    def _add_sparse(rew, d):
        rew = (rew + 1) if d < 0.5 else rew
        rew = (rew + 2) if d < 0.4 else rew
        rew = (rew + 3) if d < 0.3 else rew
        rew = (rew + 5) if d < 0.2 else rew
        rew = (rew + 8) if d < 0.05 else rew
        return rew

    # get the poses of targets in the robot's base coordinate system
    tool = get_target_pose(sim, 'base_link', 'gripperpalm')
    grip = get_target_pose(sim, 'base_link', 'CB17')
    body = get_random_target()
    gamma_1, gamma_2, gamma_3 = 1.0, 0.03, 1.0

    # reward from decreasing the distance between a gripper and a grip point - reach reward
    d = np.linalg.norm(grip[0] - tool[0])
    grip_tool_dist = d if d < 0.3 else np.square(d)
    position_rew = -gamma_1 * grip_tool_dist - gamma_2 * np.matmul(u, np.transpose(u))

    # add a sparse rewards
    position_rew = _add_sparse(position_rew, d)

    d2 = np.linalg.norm(body - grip[0])
    target_tool_dist = d2 if d2 < 0.3 else np.square(d2)
    position_rew += -gamma_3 * target_tool_dist
    position_rew = _add_sparse(position_rew, d2)

    return position_rew, d


def standarize_rewards(rewards: list):
    rewards = np.asarray(rewards)

    # standarize
    m, s = np.mean(rewards), np.sqrt(np.var(rewards))
    ret = (rewards - m) / (s + 1e-06)

    lowest = np.abs(np.min(ret))
    ret += (2 * lowest)
    ret = discount_rewards(ret)

    return np.sum(ret), np.mean(ret)
