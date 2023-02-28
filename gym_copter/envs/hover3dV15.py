'''
3D Copter-Hover class

Copyright (C) 2021 Simon D. Levy

MIT License
'''
import math

from numpy import degrees
import numpy as np

from gym_copter.envs.hover import _Hover
from gym_copter.envs.threed import _ThreeD
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


def gaussian_transform(a, sigma, x, y_offset=0):
    y = a * (np.exp(-1 * ((x ** 2) / (2 * sigma)))) + y_offset

    return y


class Hover3DV15(_Hover, _ThreeD):

    def __init__(self, obs_size=12):
        _Hover.__init__(self, obs_size, 4, max_steps=20000, out_of_bounds_penalty=100, initial_altitude=8)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

    def reset(self):
        return _Hover._reset(self)

    def _get_state(self, state):
        return state

    def _get_reward(self, status, state, d, x, y):
        position_sigma = 3
        position_amplitude = 1

        target = 0, 0, -5

        x_reward = gaussian_transform(position_amplitude, position_sigma, state[0] - target[0])
        y_reward = gaussian_transform(position_amplitude, position_sigma, state[2] - target[1])
        z_reward = gaussian_transform(position_amplitude, position_sigma, state[4] - target[2])

        reward = ((0.075 * x_reward) + (0.075 * y_reward) + (0.85 * z_reward))

        z = state[4]

        if abs(x) >= self.bounds or abs(y) >= self.bounds or abs(z) >= self.bounds or z >= 0:
            self.done = True
            reward -= self.out_of_bounds_penalty

        return reward

    def use_hud(self):
        _ThreeD.use_hud(self)

    def render(self, mode='human'):
        return _ThreeD.render(self, mode)

    def demo_pose(self, args):
        _ThreeD.demo_pose(self, args)

    def get_position_sigma(self):
        return NotImplementedError


class HoverVisual(Hover3DV15):
    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):
        Hover3DV15.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):
        result = Hover3DV15.step(self, action)

        x, y, z, phi, theta, psi = self.pose

        self.image = self.vs.getImage(x,
                                      y,
                                      max(-z, 1e-6),  # keep Z positive
                                      degrees(phi),
                                      degrees(theta),
                                      degrees(psi))

        return result

    def render(self, mode='human'):
        if self.image is not None:
            self.vs.display_image(self.image)


class HoverDVS(HoverVisual):

    def __init__(self):
        HoverVisual.__init__(self, vs=DVS(res=HoverVisual.RES))
