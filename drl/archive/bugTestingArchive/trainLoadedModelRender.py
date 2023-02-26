import numpy as np
import gym as gym
from stable_baselines3 import PPO, DDPG
import os
import pandas as pd
import time as sleep

def _heuristic(env):
    project_name = "gymCopter-Hover3DV10-z_r_tol=0.1-r_f=1000-1675863205"
    time_step = 15_000_000
    models_dir = f"models/{project_name}"
    model_path = f"{models_dir}/{time_step}.zip"

    print(f"Project Name: {project_name}\nTimeStep: {time_step}")

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False

    steps = 0
    total_reward = 0

    while not done:
        actions, values, log_probs = model.policy.forward(obs)
        state, reward, done, _ = env.step(actions)

        total_reward += reward

        print('(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (state[0], state[2], state[4], state[6], state[8], state[10], steps, reward, total_reward))
        env.render()

        sleep(1. / env.FRAMES_PER_SECOND)
        steps += 1

    print(total_reward)
    env.close()


print("Setting Names")

project_name = "gymCopter-Hover3DV10-z_r_tol=0.1-r_f=1000-1675863205"
time_step = 15_000_000
# project_name = "gymCopter-Hover3D-DDPG-1674386090"
models_dir = f"models/{project_name}"
model_path = f"{models_dir}/{time_step}.zip"

print(f"Project Name: {project_name}\nTimeStep: {time_step}")

env = gym.make("gym_copter:Hover3D-v10")
env.reset()

model = PPO.load(model_path, env=env)
model.env.reset()

logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print("Setting up envs and models")

TIMESTEPS = 5_000
iters = 0

episodes = 5

for ep in range(episodes):
    # input("Press enter in the command window to continue.....")
    env = gym.make("gym_copter:Hover3D-10")
    env.reset()

    viewer = ThreeDHoverRenderer(env,
                                 _heuristic,
                                 ())

    viewer.start()