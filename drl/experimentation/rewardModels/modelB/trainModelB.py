import gym as gym
from stable_baselines3 import PPO
import os
import time
import numpy as np


def runOneRun(position_sigma):
    total_runs = 1
    print(f"I will train {total_runs} runs")

    starting_index = 1

    # position_sigma = 10
    attitude_sigma = np.pi

    for run in range(starting_index, total_runs + starting_index):
        print(f"Training run {run}")
        project_name = f"ModelB_V1_1," \
                       f"position_sigma_{str(position_sigma).replace('.', '_')}," \
                       f"attitude_sigma_{str(round(attitude_sigma, 5)).replace('.', '_')}," \
                       f"run_{run}"
        models_dir = f"models/{project_name}"

        logdir = "logs"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        fault_magnitude = [1, 1, 1, 1]
        weights = (0.1, 0.1, 0.5, 0.1, 0.1, 0.1)

        env = gym.make("gym_copter:ModelB-v1",
                       fault_magnitude=fault_magnitude,
                       weights=weights,
                       position_sigma=position_sigma,
                       attitude_sigma=attitude_sigma,
                       )
        env.reset()

        print(f"Current run weights are {[f'{w:.4f}' for w in env.weights]}")
        print(f"Position sigma is :{position_sigma}")
        print(f"Attitude sigma is :{attitude_sigma}")

        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

        total_timesteps = 5_000_000
        timesteps = 200_000
        iterations = int(total_timesteps / timesteps)

        for j in range(1, iterations + 1):
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=project_name)
            model.save(f"{models_dir}/{timesteps * j}")

        print(f"Finished run {run}")

    print(f"Finished Training")


def main():
    # position_sigma = [10, 7, 5, 3, 2, 1.5, 1, 0.75, 0.5, 0.25]
    position_sigmas = [7]

    for p_s in position_sigmas:
        runOneRun(p_s)


if __name__ == '__main__':
    main()
