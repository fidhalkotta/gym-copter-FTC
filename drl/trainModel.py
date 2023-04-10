import numpy as np
import gym as gym
from stable_baselines3 import PPO, DDPG
import torch as th

import os
import time


def main():
    print("Setting up envs and models")

    fault_magnitude = [0.75, 1, 1, 1]

    env = gym.make("gym_copter:Hover3D-v30",
                   position_sigma=0.5, attitude_sigma=(np.pi / 5),
                   fault_magnitude=fault_magnitude
                   )
    env.reset()

    print("Setting Names")

    project_name = f"V30-RL-fault-0_75-passive-p_sigma-0_5-a_sigma-0_2-wind_power-1_0-{int(time.time())}"
    models_dir = f"models/{project_name}"

    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    # model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=logdir)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 200_000

    print("Starting Training")
    for i in range(1, 26):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=project_name)
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    print("Completed Training")


if __name__ == '__main__':
    main()
