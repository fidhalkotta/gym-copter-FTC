import gym as gym
import numpy as np
from stable_baselines3 import PPO, DDPG
from time import sleep

from gym_copter.rendering.threed import ThreeDHoverRenderer

from stable_baselines3.common.base_class import BaseAlgorithm


def _heuristic(env):
    # project_name = "gymCopter-Hover3D-PPO-1674385145"
    # project_name = "gymCopter-Hover3DV10-DDPG-z_r_tol=0.1-r_f=1000-1675925309"
    # project_name = "gymCopter-Hover3DV1-PPO-1674436885"
    # project_name = "gymCopter-Hover3DV3-PPO-1674806710"
    # project_name = "gymCopter-Hover3DV4-PPO-1674810537"
    # project_name = "gymCopter-Hover3DV5-PPO-1674812459"
    # project_name = "gymCopter-Hover3DV6-PPO-1674995181"
    # project_name = "gymCopter-Hover3DV6-PPO-1674990560"
    # project_name = "gymCopter-Hover3DV7-PPO-1674994191"
    # project_name = "gymCopter-Hover3DV8-1674996991"
    # project_name = "gymCopter-Hover3DV9-1675010997"
    # project_name = "gymCopter-Hover3DV6-sigma=0_5-1675603657"
    project_name = "gymCopter-Hover3DV10-z_r_tol=0.1-r_f=1000-1675863205"
    # project_name = "gymCopter-Hover3DV12-1675878758"
    # project_name = "gymCopter-Hover3DV13-1675922924"
    time_step = 14_000_000
    models_dir = f"models/{project_name}"
    model_path = f"{models_dir}/{time_step}.zip"

    print(f"Project Name: {project_name}\nTimeStep: {time_step}")

    print(env)

    env = BaseAlgorithm._wrap_env(env, 1, True)

    print(env)

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False

    steps = 0
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        state, reward, done, _ = env.step(action)

        total_reward += reward

        # print('(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (state[0], state[2], state[4], state[6], state[8], state[10], steps, reward, total_reward))
        env.render()

        sleep(1. / 60)
        steps += 1

    print(total_reward)
    env.close()


def main():
    episodes = 5

    for ep in range(episodes):
        # input("Press enter in the command window to continue.....")
        env = gym.make("gym_copter:Hover3D-v10")
        env.reset()

        viewer = ThreeDHoverRenderer(env,
                                     _heuristic,
                                     ())

        viewer.start()



main()