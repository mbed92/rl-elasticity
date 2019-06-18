class Env(object):
    def __init__(self, sim_start, sim_step):
        self.sim_start = sim_start
        self.sim_step = sim_step

    def get_reward(self, *args):
        raise NotImplementedError("Not implemented")

    def step(self, *args):
        raise NotImplementedError("Not implemented")

    def reset(self):
        raise NotImplementedError("Not implemented")

    def get_observations(self):
        raise NotImplementedError("Not implemented")

    def take_continuous_action(self, *args):
        raise NotImplementedError("Not implemented")

    def randomize_environment(self):
        raise NotImplementedError("Not implemented")
