'''
3D Copter-Hover class

Copyright (C) 2021 Simon D. Levy

MIT License
'''
import math
import random

from numpy import degrees
import numpy as np

from gym_copter.envs.hover import _Hover
from gym_copter.envs.threed import _ThreeD
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


def gaussian_transform(a, sigma, x, y_offset=0):
    y = a * (np.exp(-1 * ((x ** 2) / (2 * sigma)))) + y_offset

    return y


class Hover3DV22(_Hover, _ThreeD):

    def __init__(self, obs_size=12):
        _Hover.__init__(self, obs_size, 4, max_steps=20000, out_of_bounds_penalty=100, initial_altitude=8, initial_random_position=False)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

        # For generating plots
        self.plot = False

        self.fault_magnitude = [0.5, 1, 1, 1]

    def set_fault_state(self, fault_state):
        if fault_state:
            self.fault_map = self.fault_magnitude
        else:
            self.fault_map = [1, 1, 1, 1]


    def reset(self):
        return _Hover._reset(self)

    def _get_state(self, state):
        return state

    def _get_reward(self, status, state, d, x, y):
        position_sigma = 3
        self.position_sigma = position_sigma
        angle_sigma = np.pi / 5

        position_amplitude = 1
        angle_amplitude = 1

        target = 0, 0, -5

        x_reward = gaussian_transform(position_amplitude, position_sigma, state[0] - target[0])
        y_reward = gaussian_transform(position_amplitude, position_sigma, state[2] - target[1])
        z_reward = gaussian_transform(position_amplitude, position_sigma, state[4] - target[2])

        phi_reward = gaussian_transform(angle_amplitude, angle_sigma, state[6])
        theta_reward = gaussian_transform(angle_amplitude, angle_sigma, state[8])
        psi_reward = gaussian_transform(angle_amplitude, angle_sigma, state[10])

        reward = (0.11 * x_reward) + (0.11 * y_reward) + (0.485 * z_reward) + (0.11 * phi_reward) + (0.11 * theta_reward) + (0.075 * psi_reward)

        return reward

    def use_hud(self):
        _ThreeD.use_hud(self)

    def render(self, mode='human'):
        return _ThreeD.render(self, mode)

    def demo_pose(self, args):
        _ThreeD.demo_pose(self, args)

    def get_position_sigma(self):
        return NotImplementedError

    # Don't think this does anything
    # def fault_action_transformer(self, action):
    #     for i in range(len(self.fault_map)):
    #         action[i] *= self.fault_map[i]
    #
    #     return action

    def handle_fault_injection(self):
        self.fault_map = self.fault_magnitude
        self.viewer.flip_fault_state()

    def handle_fault_removal(self):
        self.fault_map = [1, 1, 1, 1]
        self.viewer.flip_fault_state()

    def create_plots(self):
        self.plot = True


class HoverVisual(Hover3DV22):
    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):
        Hover3DV22.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):
        result = Hover3DV22.step(self, action)

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
