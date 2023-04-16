import gym as gym
from stable_baselines3 import PPO, DDPG
from time import sleep

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

from gym_copter.rendering.threed import ThreeDHoverRenderer


def _heuristic(env):
    # project_name = "ModelB_V1_1,position_sigma_7_5,attitude_sigma_3_14159,run_1"  # 2_400_000 - crashes after around 2000 timesteps,
    # project_name = "ModelB_V1_1,position_sigma_3,attitude_sigma_3_14159,run_1"  # 3_000_000 - HITS 20,000, oscillates and kinda orbits in a patterns around the target below the point
    # project_name = "ModelB_V1_1,position_sigma_2,attitude_sigma_3_14159,run_1"  # 2_800_000 - good sometimes
    # project_name = "ModelB_V1_1,position_sigma_2,attitude_sigma_1_5708,run_1"  # 4_200_000 - really goood sometimes. like hits 20,000, very stable.
    # project_name = "ModelB_V1_1,position_sigma_1,attitude_sigma_1_0472,run_1"  # 2_200_000 - think its pretty good? bit wibbly but wil not fail, aACTUALLY JUST CRASHED AT 6000
    # project_name = "ModelB_V1_1,position_sigma_0_5,attitude_sigma_0_62832,run_1"  # 5_000_000   - highest peak in reward, decent sometimes
    # project_name = "ModelB_V1_1,position_sigma_3,attitude_sigma_1_5708,run_1"  # 2_400_000   -

    project_name = "ModelB_V1_1,position_sigma_0_25,attitude_sigma_0_62832,run_1"  # 2_200_000 - BEST BY FAR - works with 0.85 fault pretty much always, and with 0.75 works 1 out of 3 times


    time_step = 2_000_000
    models_dir = f"models/{project_name}"
    model_path = f"{models_dir}/{time_step}.zip"

    save_data = False
    save_data_steps_limit = 6_000
    save_data_file_name = f"data/RL-{project_name}-{time_step}.csv"

    print(f"Project Name: {project_name}\nTimeStep: {time_step}")

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    # sleep(2)
    env.set_fault_state(True)
    done = False

    steps = 0
    real_time = 0 / env.FRAMES_PER_SECOND

    if save_data:
        f = open(save_data_file_name, "w")

        states_data = pd.DataFrame(columns=["time_step", "real_time",
                                            "x", "y", "z", "phi", "theta", "psi",
                                            "reward", "total_reward"])

        new_df = pd.DataFrame([[steps, real_time, obs[0], obs[2], obs[4], obs[6], obs[8], obs[10], 0, 0]],
                              columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi", "reward", "total_reward"])
        states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

    print(env.fault_map)
    while not done:
        action, _ = model.predict(obs)

        obs, reward, done, _ = env.step(action)

        print(
            '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (
            obs[0], obs[2], obs[4], obs[6], obs[8], obs[10], steps, reward, env.total_reward))
        env.render()

        sleep(1. / env.FRAMES_PER_SECOND)
        steps += 1
        real_time = steps / env.FRAMES_PER_SECOND

        if save_data:
            new_df = pd.DataFrame([[steps, real_time, obs[0] , obs[2], obs[4], obs[6], obs[8], obs[10], reward, env.total_reward]],
                                  columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi", "reward", "total_reward"])
            states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

            if steps > save_data_steps_limit:
                done = True

    print(env.total_reward)
    print(env.fault_map)
    env.close()

    if save_data:
        states_data.to_csv(save_data_file_name, index=False)


def main():
    episodes = 3

    for ep in range(episodes):
        fault_magnitude = [0.65, 1, 1, 1]
        weights = (0.1, 0.1, 0.5, 0.1, 0.1, 0.1)

        env = gym.make("gym_copter:ModelB-v1",
                       fault_magnitude=fault_magnitude,
                       weights=weights,
                       )
        env.reset()

        viewer = ThreeDHoverRenderer(env,
                                     _heuristic,
                                     (),
                                     outfile=None)

        viewer.start()


if __name__ == '__main__':
    main()

