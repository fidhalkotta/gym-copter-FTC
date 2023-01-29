import numpy as np
import gym as gym
from stable_baselines3 import PPO, DDPG
import os
import time

print("Setting Names")

project_name = f"gymCopter-Hover3DV9-{int(time.time())}"
models_dir = f"models/{project_name}"

logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print("Setting up envs and models")

env = gym.make("gym_copter:Hover3D-v9")
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 200_000
iters = 0

print("Starting Training")
for i in range(1, 51):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=project_name)
    model.save(f"{models_dir}/{TIMESTEPS * i}")

print("Completed Training")