import numpy as np
import gym as gym
from stable_baselines3 import PPO, DDPG
import os
import pandas as pd
import time

print("Setting Names")

project_name = "gymCopter-Hover3DV17-1677488099"
time_step = 5_000_000
# project_name = "gymCopter-Hover3D-DDPG-1674386090"
models_dir = f"models/{project_name}"
model_path = f"{models_dir}/{time_step}.zip"

print(f"Project Name: {project_name}\nTimeStep: {time_step}")

env = gym.make("gym_copter:Hover3D-v17")
env.reset()

model = PPO.load(model_path, env=env)
model.env.reset()

# policy = model.get_parameters()
#
# piNet = policy["policy"]["mlp_extractor.policy_net.0.weight"]
#
# t_np = piNet.numpy()  # convert to Numpy array
# df = pd.DataFrame(t_np)  # convert to a dataframe
# csvName = f"TRAIN_timeStep={model._total_timesteps}_policy0.csv"
#
# df.to_csv(csvName)  # save to file
#
# # Then, to reload:
# df = pd.read_csv(csvName)

logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print("Setting up envs and models")

TIMESTEPS = 200_000
iters = 0

print("Restarting Training")
for i in range(1, 26):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=project_name)
    model.save(f"{models_dir}/{time_step + (TIMESTEPS * i)}")

print("Completed Training")