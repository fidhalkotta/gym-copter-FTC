import gym as gym
from stable_baselines3 import PPO
import os
import time
import numpy as np


def main():
    total_runs = 1
    print(f"I will train {total_runs} runs")

    starting_index = 4

    for run in range(starting_index, total_runs + starting_index):
        print(f"Training run {run}")
        project_name = f"ModelA_V2_1,run_{run}"
        models_dir = f"models/{project_name}"

        logdir = "logs"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        fault_magnitude = [1, 1, 1, 1]
        weights = (0.1, 0.1, 0.5, 0.1, 0.1, 0.1)

        env = gym.make("gym_copter:ModelA-v2",
                       fault_magnitude=fault_magnitude,
                       weights=weights,
                       )
        env.reset()

        print(f"Current run weights are {[f'{w:.4f}' for w in env.weights]}")

        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

        total_timesteps = 5_000_000
        timesteps = 200_000
        iterations = int(total_timesteps / timesteps)

        for j in range(1, iterations + 1):
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=project_name)
            model.save(f"{models_dir}/{timesteps * j}")

        print(f"Finished run {run}")

    print(f"Finished Training")


if __name__ == '__main__':
    main()
