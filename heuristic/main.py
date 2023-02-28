#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo support

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np

import gym

from gym_copter.cmdline import (make_parser, make_parser_3d,
                                wrap, parse_view_angles)


def _demo_heuristic(env, fun, pidcontrollers,
                    seed=None, csvfilename=None, nopid=False):

    env.seed(seed)
    np.random.seed(seed)

    steps = 0
    state = env.reset()

    dt = 1. / env.FRAMES_PER_SECOND

    actsize = env.action_space.shape[0]

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, actsize+1)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    while True:

        action = np.zeros(actsize) if nopid else fun(state, pidcontrollers)

        action = list(action)

        if env.total_reward > 2000:
            action[0] *= 0.9

        state, reward, done, _ = env.step(action)

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * actsize) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        steps += 1

        print(
            '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (state[0], state[2], state[4], state[6], state[8], state[10], steps, reward, env.total_reward))

        if done:
            break

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
