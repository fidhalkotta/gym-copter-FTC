import gym as gym
from stable_baselines3 import PPO, DDPG, TD3
import os
import time


def main():
    total_runs = 3
    print(f"I will train {total_runs} runs")

    for run in range(1, total_runs + 1):
        print(f"Training run {run}")
        project_name = f"random_test,TD3,run_{run}"
        models_dir = f"models/{project_name}"

        logdir = "logs"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        env = gym.make('LunarLander-v2', continuous=True, enable_wind=True)
        env.reset()

        model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

        total_timesteps = 500_000
        timesteps = 50_000
        iterations = int(total_timesteps / timesteps)

        for j in range(1, iterations + 1):
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=project_name)
            model.save(f"{models_dir}/{timesteps * j}")

        print(f"Finished run {run}")


if __name__ == '__main__':
    main()
