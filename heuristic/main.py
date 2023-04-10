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


def _demo_heuristic(env, fun, pidcontrollers,
                    seed=None, csvfilename=None, nopid=False):

    project_name = "gymCopter-Hover3DV28-PID"

    save_data = False
    save_data_steps_limit = 5_000
    save_data_file_name = f"../drl/data/{project_name}-nominal.csv"

    env.seed(seed)
    np.random.seed(seed)

    steps = 0
    real_time = 0 / env.FRAMES_PER_SECOND
    obs = env.reset()
    env.set_fault_state(True)
    done = False
    flip = False

    dt = 1. / env.FRAMES_PER_SECOND

    actsize = env.action_space.shape[0]

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, actsize+1)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    if save_data:
        f = open(save_data_file_name, "w")

        states_data = pd.DataFrame(columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi"])

        new_df = pd.DataFrame([[steps, real_time, obs[0], obs[2], obs[4], obs[6], obs[8], obs[10]]],
                              columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi"])
        states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

    while not done:
        action = np.zeros(actsize) if nopid else fun(obs, pidcontrollers)
        action = list(action)

        # if env.total_reward > 500:
        #     if not flip:
        #         env.handle_fault_injection()
        #         flip = True

        obs, reward, done, _ = env.step(action)

        # if csvfile is not None:
        #
        #     csvfile.write('%f' % (dt * steps))
        #
        #     csvfile.write((',%f' * actsize) % tuple(action))
        #
        #     csvfile.write(((',%f' * len(obs)) + '\n') % tuple(obs))

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        print(
            '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (obs[0], obs[2], obs[4], obs[6], obs[8], obs[10], steps, reward, env.total_reward))

        steps += 1
        real_time = steps / env.FRAMES_PER_SECOND

        if save_data:
            new_df = pd.DataFrame([[steps, real_time, obs[0], obs[2], obs[4], obs[6], obs[8], obs[10]]],
                                  columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi"])
            states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

            if steps > save_data_steps_limit:
                done = True


    if save_data:
        states_data.to_csv(save_data_file_name, index=False)

    env.close()

    if csvfile is not None:
        csvfile.close()


def demo(envname, heuristic, pidcontrollers):

    parser = make_parser()

    args = parser.parse_args()

    env = gym.make(envname)

    if args.movie:
        env = wrap(args, env)

    _demo_heuristic(env, heuristic, pidcontrollers,
                    seed=args.seed, csvfilename=args.csvfilename,
                    nopid=args.nopid)

    env.close()


def demo3d(envname, heuristic, pidcontrollers, renderer):

    env = gym.make(envname)

    parser = make_parser_3d()

    args = parser.parse_args()

    if args.hud:

        env = wrap(args, env)

        env.use_hud()

        _demo_heuristic(env, heuristic, pidcontrollers,
                        args.seed, args.csvfilename, args.nopid)

    else:

        viewer = renderer(env,
                          _demo_heuristic,
                          (heuristic, pidcontrollers,
                           args.seed, args.csvfilename, args.nopid),
                          viewangles=parse_view_angles(args),
                          outfile='movie.mp4' if args.movie else None)

        viewer.start()
