import gym as gym
from stable_baselines3 import PPO
import os
import time
import numpy as np


def runOneRun(fault_magnitude):
    total_runs = 2
    print(f"I will train {total_runs} runs")
    print(f"Current faults are {[f'{f:.4f}' for f in fault_magnitude]}")

    starting_index = 2

    for run in range(starting_index, total_runs + starting_index):
        print(f"Training run {run}")
        project_name = f"ModelB_V2_1," \
                       f"fm0_{str(fault_magnitude[0]).replace('.', '_')}," \
                       f"fm1_{str(fault_magnitude[1]).replace('.', '_')}," \
                       f"fm2_{str(fault_magnitude[2]).replace('.', '_')}," \
                       f"fm3_{str(fault_magnitude[3]).replace('.', '_')}," \
                       f"run_{run}"
        models_dir = f"models/{project_name}"

        logdir = "logs"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        env = gym.make("gym_copter:ModelB-v2",
                       fault_magnitude=fault_magnitude,
                       )
        env.reset()

        print(f"Current env-run faults are {[f'{f:.4f}' for f in env.fault_magnitude]}")

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
    fault_magnitude = [0.75, 1, 1, 1]

    runOneRun(fault_magnitude)


if __name__ == '__main__':
    main()
