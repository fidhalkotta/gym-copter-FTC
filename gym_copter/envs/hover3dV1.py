'''
3D Copter-Hover class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from numpy import degrees
import numpy as np

from gym_copter.envs.hover import _Hover
from gym_copter.envs.threed import _ThreeD
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


class Hover3DV1(_Hover, _ThreeD):

    def __init__(self, obs_size=12):

        _Hover.__init__(self, obs_size, 4)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

    def reset(self):

        return _Hover._reset(self)

    def _get_state(self, state):

        return state

    def _get_reward(self, status, state, d, x, y):
        xyz_penalty_factor = 10

        # Get penalty based on state and motors
        shaping = -(xyz_penalty_factor * np.sqrt(np.sum(state[0:3] ** 2)))

        reward = ((shaping - self.prev_shaping)
                  if (self.prev_shaping is not None)
                  else 0)

        self.prev_shaping = shaping

        return reward

    def use_hud(self):

        _ThreeD.use_hud(self)

    def render(self, mode='human'):
        return _ThreeD.render(self, mode)

    def demo_pose(self, args):

        _ThreeD.demo_pose(self, args)


class HoverVisual(Hover3DV1):

    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):

        Hover3DV1.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):

        result = Hover3DV1.step(self, action)

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
