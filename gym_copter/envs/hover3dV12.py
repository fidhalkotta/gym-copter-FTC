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


class Hover3DV12(_Hover, _ThreeD):

    def __init__(self, obs_size=12):
        _Hover.__init__(self, obs_size, 4, max_steps=20000, out_of_bounds_penalty=0, initial_altitude=8)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

    def reset(self):
        return _Hover._reset(self)

    def _get_state(self, state):
        return state

    def _get_reward(self, status, state, d, x, y):
        target = 0, 0, -5
        z_r = state[4] - target[2]

        r_0 = -1

        if abs(z_r) < 0.5:
            r_0 = 1

        nu_normalised = math.sqrt((state[1]**2) + (state[3]**2) + (state[5]**2))
        omega_normalised = math.sqrt((state[7]**2) + (state[9]**2) + (state[11]**2))

        reward = r_0 - nu_normalised - omega_normalised

        return reward

    def use_hud(self):
        _ThreeD.use_hud(self)

    def render(self, mode='human'):
        return _ThreeD.render(self, mode)

    def demo_pose(self, args):
        _ThreeD.demo_pose(self, args)

    def get_position_sigma(self):
        return NotImplementedError


class HoverVisual(Hover3DV12):
    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):
        Hover3DV12.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):
        result = Hover3DV12.step(self, action)

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
