import gym as gym
import numpy as np
from stable_baselines3 import PPO, DDPG
from time import sleep

from gym_copter.rendering.threed import ThreeDHoverRenderer


def _heuristic(env):
    # project_name = "gymCopter-Hover3D-PPO-1674385145"
    # project_name = "gymCopter-Hover3DV1-PPO-1674436885"
    # project_name = "gymCopter-Hover3DV3-PPO-1674806710"
    # project_name = "gymCopter-Hover3DV4-PPO-1674810537"
    project_name = "gymCopter-Hover3DV5-PPO-1674812459"
    time_step = 1_000_000
    # project_name = "gymCopter-Hover3D-DDPG-1674386090"
    models_dir = f"models/{project_name}"
    model_path = f"{models_dir}/{time_step}.zip"

    print(f"Project Name: {project_name}\nTimeStep: {time_step}")

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False

    steps = 0

    while not done:
        action, _states = model.predict(obs)
        state, reward, done, _ = env.step(action)

        print('steps = %04d    current_reward = %+0.2f' % (steps, reward))
        env.render()

        sleep(1. / env.FRAMES_PER_SECOND)
        steps += 1

    env.close()


def main():
    episodes = 5

    for ep in range(episodes):
        env = gym.make("gym_copter:Hover3D-v5")
        env.reset()

        viewer = ThreeDHoverRenderer(env,
                                     _heuristic,
                                     ())

        viewer.start()

main()