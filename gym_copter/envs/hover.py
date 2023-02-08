'''
Superclass for 2D and 3D copter hover

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
from gym_copter.envs.task import _Task


class _Hover(_Task):

    def __init__(self, observation_size, action_size, max_steps=1000):

        _Task.__init__(self, observation_size, action_size, max_steps=max_steps)

    def _get_reward(self, status, state, d, x, y):

        # Simple reward for each step we complete
        return 1

    def get_position_sigma(self):
        return self.position_sigma