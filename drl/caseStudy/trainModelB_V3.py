import gym as gym
from stable_baselines3 import PPO
import os
import time
import numpy as np


def runOneRun(fault_cases, f_i):
    total_runs = 1
    print(f"I will train {total_runs} runs")
    print(f"Current faults cases are\n {fault_cases}")

    starting_index = 3

    for run in range(starting_index, total_runs + starting_index):
        print(f"Training run {run}")
        project_name = f"ModelB_V3_1," \
                       f"fmi_{str(f_i).replace('.', '_')}," \
                       f"run_{run}"
        models_dir = f"models/{project_name}"

        logdir = "logs"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        env = gym.make("gym_copter:ModelB-v3",
                       fault_cases=fault_cases,
                       ) 
        env.reset()

        print(f"Current faults cases are\n {env.fault_cases}")

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
    f_i = 0.65

    fault_cases = [
            [1, 1, 1, 1],
            [f_i, 1, 1, 1],
            [1, f_i, 1, 1],
            [1, 1, f_i, 1],
            [1, 1, 1, f_i],
    ]

    runOneRun(fault_cases, f_i)


if __name__ == '__main__':
    main()
