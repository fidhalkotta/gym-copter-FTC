import gym
import numpy as np
from time import sleep


from gym_copter.rendering.threed import ThreeDHoverRenderer

def _heuristic(env):

    seed = 42

    env.seed(seed)
    np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    dt = 1. / env.FRAMES_PER_SECOND

    while True:
        state, reward, done, _ = env.step(env.action_space.sample())
        total_reward += reward

        env.render()

        sleep(1. / env.FRAMES_PER_SECOND)

        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    current_reward = %+0.2f    total_reward = %+0.2f' %
                  (steps, reward, total_reward))

        if done:
            break

    env.close()


def main():

    for _ in range(3):
        env = gym.make("gym_copter:Hover3D-v16")

        viewer = ThreeDHoverRenderer(env,
                          _heuristic,
                          ())

        viewer.start()


main()
