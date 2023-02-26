import gym as gym
import numpy as np
from stable_baselines3 import PPO, DDPG
from time import sleep
import pandas as pd

from gym_copter.rendering.threed import ThreeDHoverRenderer

def main():
    episodes = 5

    for ep in range(episodes):
        # input("Press enter in the command window to continue.....")
        env = gym.make("gym_copter:Hover3D-v10")
        env.reset()

        project_name = "gymCopter-Hover3DV10-z_r_tol=0.1-r_f=1000-1675863205"
        time_step = 15_000_000
        models_dir = f"models/{project_name}"
        model_path = f"{models_dir}/{time_step}.zip"

        print(f"Project Name: {project_name}\nTimeStep: {time_step}")

        model = PPO.load(model_path, env=env)

        policy = model.get_parameters()

        piNet = policy["policy"]["mlp_extractor.policy_net.0.weight"]

        t_np = piNet.numpy()  # convert to Numpy array
        df = pd.DataFrame(t_np)  # convert to a dataframe
        csvName = f"LOAD_timeStep={model._total_timesteps}_policy0.csv"

        df.to_csv(csvName)  # save to file

        # Then, to reload:
        df = pd.read_csv(csvName)

        obs = env.reset()
        done = False

        steps = 0
        total_reward = 0

        while not done:
            action, _states = model.predict(obs)
            state, reward, done, _ = env.step(action)

            total_reward += reward

            # print(
            #     '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (
            #     state[0], state[2], state[4], state[6], state[8], state[10], steps, reward, total_reward))

            sleep(1. / env.FRAMES_PER_SECOND)
            steps += 1

        print(total_reward)
        env.close()



main()