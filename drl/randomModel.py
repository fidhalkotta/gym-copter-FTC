import gym
import numpy as np
from time import sleep


from gym_copter.rendering.threed import ThreeDHoverRenderer


def main():
    for _ in range(5):
        env = gym.make("gym_copter:Hover3D-v0")

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
                print('steps =  %04d    total_reward = %+0.2f' %
                      (steps, total_reward))

            if done:
                break

        env.close()


main()
