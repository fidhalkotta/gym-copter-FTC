import gym
import numpy as np
from time import sleep


from gym_copter.rendering.threed import ThreeDHoverRenderer


def main():
    for _ in range(1):
        env = gym.make("gym_copter:Hover3D-v10")

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


            print(
                '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (
                state[0], state[2], state[4], state[6], state[8], state[10], steps, reward, total_reward))

            if done:
                break

        env.close()


main()
