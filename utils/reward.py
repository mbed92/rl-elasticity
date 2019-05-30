from .interactions import *


def get_sparse_reward(sim):
    tool = get_target_pose(sim, 'base_link', 'gripperpalm')
    grip = get_target_pose(sim, 'base_link', 'CB13')
    body = get_random_target()
    reward = -1.0

    d1 = np.linalg.norm(grip[0] - tool[0])
    d2 = np.linalg.norm(body - grip[0])

    if d1 < 0.25:
        reward = 100.0
        if d1 < 0.08:
            reward = 200.0
            # if d2 < 0.15:
            #     reward = 300.0
            #     if d2 < 0.05:
            #         reward = 400.0

    return reward, d1, d2


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


def discount_rewards(r, gamma=0.98):
    discounted_r = np.zeros_like(r)
    r = np.asarray(r)
    for t in range(0, r.size):
        discounted_r[t] = np.power(gamma, r.size - t - 1) + r[t] if r[t] > 0 else r[t]
    return discounted_r
