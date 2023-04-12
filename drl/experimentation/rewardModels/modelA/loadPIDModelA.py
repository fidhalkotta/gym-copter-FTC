#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo support

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np
import pandas as pd

import gym

from gym_copter.cmdline import (make_parser, make_parser_3d,
                                wrap, parse_view_angles)
from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import AltitudeHoldPidController
from gym_copter.rendering.threed import ThreeDHoverRenderer



def heuristic(state, pidcontrollers):
    '''
    PID controller
    '''
    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, _, dpsi = state

    (roll_rate_pid,
     pitch_rate_pid,
     yaw_rate_pid,
     x_poshold_pid,
     y_poshold_pid,
     althold_pid) = pidcontrollers

    roll_rate_todo = roll_rate_pid.getDemand(dphi)
    y_pos_todo = x_poshold_pid.getDemand(y, dy)

    pitch_rate_todo = pitch_rate_pid.getDemand(-dtheta)
    x_pos_todo = y_poshold_pid.getDemand(x, dx)

    roll_todo = roll_rate_todo + y_pos_todo
    pitch_todo = pitch_rate_todo + x_pos_todo
    yaw_todo = yaw_rate_pid.getDemand(-dpsi)

    hover_todo = althold_pid.getDemand(z, dz)

    t, r, p, y = (hover_todo+1)/2, roll_todo, pitch_todo, yaw_todo

    # Use mixer to set motors
    return t-r-p-y, t+r+p-y, t+r-p+y, t-r+p+y


def _demo_heuristic(env, fun, pidcontrollers):

    project_name = "ModelA_V1_2"

    save_data = False
    save_data_steps_limit = 1_000
    save_data_file_name = f"data/PID-{project_name}.csv"

    print(f"Project Name: {project_name}\nTimeStep: N/A")

    obs = env.reset()
    env.set_fault_state(False)
    done = False

    steps = 0
    real_time = 0 / env.FRAMES_PER_SECOND

    if save_data:
        f = open(save_data_file_name, "w")

        states_data = pd.DataFrame(columns=["time_step", "real_time",
                                            "x", "y", "z", "phi", "theta", "psi",
                                            "reward", "total_reward"])

        new_df = pd.DataFrame([[steps, real_time, obs[0], obs[2], obs[4], obs[6], obs[8], obs[10], 0, 0]],
                              columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi", "reward", "total_reward"])
        states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

    print(env.fault_map)
    while not done:
        action = fun(obs, pidcontrollers)
        action = list(action)

        obs, reward, done, _ = env.step(action)

        print(
            '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (
            obs[0], obs[2], obs[4], obs[6], obs[8], obs[10], steps, reward, env.total_reward))
        env.render()

        sleep(1./env.FRAMES_PER_SECOND)
        steps += 1
        real_time = steps / env.FRAMES_PER_SECOND


        if save_data:
            new_df = pd.DataFrame([[steps, real_time, obs[0] , obs[2], obs[4], obs[6], obs[8], obs[10], reward, env.total_reward]],
                                  columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi", "reward", "total_reward"])
            states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

            if steps > save_data_steps_limit:
                done = True

    print(env.total_reward)
    print(env.fault_map)
    env.close()

    if save_data:
        states_data.to_csv(save_data_file_name, index=False)



def demo(envname, heuristic, pidcontrollers):

    parser = make_parser()

    args = parser.parse_args()

    env = gym.make(envname)

    if args.movie:
        env = wrap(args, env)

    _demo_heuristic(env, heuristic, pidcontrollers)

    env.close()


def demo3d(envname, heuristic, pidcontrollers, renderer):

    env = gym.make(envname)

    parser = make_parser_3d()

    args = parser.parse_args()

    if args.hud:

        env = wrap(args, env)

        env.use_hud()

        _demo_heuristic(env, heuristic, pidcontrollers)

    else:

        viewer = renderer(env,
                          _demo_heuristic,
                          (heuristic, pidcontrollers),
                          viewangles=parse_view_angles(args),
                          outfile='movie.mp4' if args.movie else None)

        viewer.start()



def main():
    pidcontrollers = (
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      PositionHoldPidController(),
                      PositionHoldPidController(),
                      AltitudeHoldPidController()
                     )

    demo3d('gym_copter:ModelA-v1', heuristic,
           pidcontrollers, ThreeDHoverRenderer)


if __name__ == '__main__':
    main()