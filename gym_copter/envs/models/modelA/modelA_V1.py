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


def _inertialToBody(inertial, rotation):
    cph, cth, cps, sph, sth, sps = _sincos(rotation)

    R = [[cps * cth, cth * sps, -sth],
         [cps * sph * sth - cph * sps, cph * cps + sph * sps * sth, cth * sph],
         [sph * sps + cph * cps * sth, cph * sps * sth - cps * sph, cph * cth]]

    return np.dot(R, inertial)


def attitudeDotToAngularVelocity(attitudeDot, rotation):
    cph, cth, cps, sph, sth, sps = _sincos(rotation)

    R = [
        [1, 0, -sth],
        [0, cph, sph * cth],
        [0, -sph, cph * cth]
    ]

    return np.dot(R, attitudeDot)


def _sincos(angles):
    phi, the, psi = angles

    cph = np.cos(phi)
    cth = np.cos(the)
    cps = np.cos(psi)
    sph = np.sin(phi)
    sth = np.sin(the)
    sps = np.sin(psi)

    return cph, cth, cps, sph, sth, sps


class ModelA_V1(_Hover, _ThreeD):
    def __init__(self, obs_size=12, fault_magnitude=None, weights=None, enable_passive_faults=False):
        _Hover.__init__(self, obs_size, 4, max_steps=20000, out_of_bounds_penalty=5000, initial_altitude=8,
                        initial_random_position=False,
                        enable_wind=True)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

        # For generating plots
        self.plot = False

        if fault_magnitude is None:
            fault_magnitude = [1, 1, 1, 1]

        self.enable_passive_faults = enable_passive_faults

        if weights is None:
            weights = (1/6, 1/6, 1/6, 1/6, 1/6, 1/6)

        self.fault_magnitude = fault_magnitude
        self.weights = weights


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
        attitude = (state[6], state[8], state[10])
        attitude_dot = (state[7], state[9], state[11])

        # Use transformation matrix to get p,q,r values from attitude
        angular_velocity = attitudeDotToAngularVelocity(attitude_dot, attitude)

        target = 0, 0, -5

        error = [
            state[0] - target[0],
            state[2] - target[1],
            state[4] - target[2],
            angular_velocity[0],
            angular_velocity[1],
            angular_velocity[2],
        ]

        reward = 0

        # Minus error squared multiplied by respective weight
        for i, w in enumerate(self.weights):
            reward -= w*(error[i] ** 2)

        return reward

    def use_hud(self):
        _ThreeD.use_hud(self)

    def render(self, mode='human'):
        return _ThreeD.render(self, mode)

    def demo_pose(self, args):
        _ThreeD.demo_pose(self, args)

    def get_position_sigma(self):
        return NotImplementedError

    def handle_fault_injection(self):
        self.fault_map = self.fault_magnitude
        self.viewer.flip_fault_state()

    def handle_fault_removal(self):
        self.fault_map = [1, 1, 1, 1]
        self.viewer.flip_fault_state()

    def create_plots(self):
        self.plot = True


class HoverVisual(ModelA_V1):
    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):
        ModelA_V1.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):
        result = ModelA_V1.step(self, action)

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
