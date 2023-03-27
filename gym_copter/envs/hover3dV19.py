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


class Hover3DV19(_Hover, _ThreeD):

    def __init__(self, obs_size=12):
        self.r_f = 5

        _Hover.__init__(self, obs_size, 4, max_steps=20000, out_of_bounds_penalty=100, initial_altitude=8, initial_random_position=False)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

        # For generating plots
        self.plot = False

    def reset(self):
        return _Hover._reset(self)

    def _get_state(self, state):
        return state

    def _get_reward(self, status, state, d, x, y):

        w = 0.1, 0.1, 0.6, 0.1, 0.1

        target = 0, 0, -5

        x_r = state[0] - target[0]
        y_r = state[2] - target[1]
        z_r = state[4] - target[2]

        phi_r = state[6]
        theta_r = state[8]

        c = - (w[0] * (x_r**2)) - (w[1] * (y_r**2)) - (w[2] * (z_r**2)) - (w[3] * (phi_r**2)) - (w[4] * (theta_r**2))

        reward = c

        if math.sqrt((x_r**2 + y_r**2 + z_r**2)) < 0.1:
            reward += self.r_f

        # print(f"envReward={reward}")

        return reward

    def use_hud(self):
        _ThreeD.use_hud(self)

    def render(self, mode='human'):
        return _ThreeD.render(self, mode)

    def demo_pose(self, args):
        _ThreeD.demo_pose(self, args)

    def get_position_sigma(self):
        return NotImplementedError

    def fault_action_transformer(self, action):
        for i in range(len(self.fault_map)):
            action[i] *= self.fault_map[i]

        return action

    def handle_fault_injection(self):
        self.fault_map = [0.5, 1, 1, 1]
        self.viewer.flip_fault_state()

    def handle_fault_removal(self):
        self.fault_map = [1, 1, 1, 1]
        self.viewer.flip_fault_state()

    def create_plots(self):
        self.plot = True


class HoverVisual(Hover3DV19):
    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):
        Hover3DV19.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):
        result = Hover3DV19.step(self, action)

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
