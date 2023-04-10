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
    # project_name = "gymCopter-Hover3D-PPO-1674385145"
    # project_name = "gymCopter-Hover3DV10-DDPG-z_r_tol=0.1-r_f=1000-1675925309"
    # project_name = "gymCopter-Hover3DV1-PPO-1674436885"
    # project_name = "gymCopter-Hover3DV3-PPO-1674806710"
    # project_name = "gymCopter-Hover3DV4-PPO-1674810537"
    # project_name = "gymCopter-Hover3DV5-PPO-1674812459"
    # project_name = "gymCopter-Hover3DV6-PPO-1674995181"
    # project_name = "gymCopter-Hover3DV6-PPO-1674990560"
    # project_name = "gymCopter-Hover3DV7-PPO-1674994191"
    # project_name = "gymCopter-Hover3DV8-1674996991"
    # project_name = "gymCopter-Hover3DV9-sigma=0_5-net_arch=32_32-1675675609"
    # project_name = "gymCopter-Hover3DV6-sigma=0_5-1675603657"
    # project_name = "gymCopter-Hover3DV10-z_r_tol=0.1-r_f=1000-1675863205"
    # project_name = "gymCopter-Hover3DV10-z_r_tol=0.1-r_f=1000-ent_coef=0.0001-1675943154"
    # project_name = "gymCopter-Hover3DV12-1675878758"
    # project_name = "gymCopter-Hover3DV13-1675922924"
    # project_name = "gymCopter-Hover3DV14-1677459385"
    # project_name = "gymCopter-Hover3DV16-1677483190"
    # project_name = "gymCopter-Hover3DV17-1677488099"
    # project_name = "gymCopter-Hover3DV18-initial_random_position=False-1677490451"
    # project_name = "gymCopter-Hover3DV19-irp=False-1678092587"
    # project_name = "gymCopter-Hover3DV20-irp=False-1678915917"
    # project_name = "gymCopter-Hover3DV21-irp=False-1678922223"
    # project_name = "gymCopter-Hover3DV22-irp=False-1678954838"
    # project_name = "gymCopter-Hover3DV23-irp=False-1678962852"
    # project_name = "gymCopter-Hover3DV24-irp=False-1679299652"
    # project_name = "gymCopter-Hover3DV25-irp=False-1679300804"
    # project_name = "gymCopter-Hover3DV26-fault-0_75-1679885987"
    # project_name = "gymCopter-Hover3DV26-fault-0_75-passive-1679889220"
    # project_name = "gymCopter-Hover3DV27-faultless-1679901288"

    # VERY GOOD ALWAYS FAULTY ONE BELOW
    # project_name = "gymCopter-Hover3DV28-fault-0_75-active-p_sigma-4-a_sigma-0_2-1680332126"

    # project_name = "gymCopter-Hover3DV28-fault-0_75-active-p_sigma-0_5-a_sigma-0_2-1680333567" # 3_000_000

    # Very very good passive faulty 0.75 below
    # project_name = "gymCopter-Hover3DV28-fault-0_75-passive-p_sigma-0_5-a_sigma-0_2-1680339074"  # 1_600_00

    # project_name = "gymCopter-Hover3DV28-fault-0_6-passive-p_sigma-0_5-a_sigma-0_2-1680344215"
    # project_name = "gymCopter-Hover3DV28-fault-0_5-passive-p_sigma-0_5-a_sigma-0_2-1680352502"
    project_name = "V29-RL-fault-0_75-passive-p_sigma-0_5-a_sigma-0_2-wind_power-1_0-1680783457"


    time_step = 2_200_000
    models_dir = f"models/{project_name}"
    model_path = f"{models_dir}/{time_step}.zip"

    save_data = False
    save_data_steps_limit = 5_000
    save_data_file_name = f"data/{project_name}-nominal.csv"

    print(f"Project Name: {project_name}\nTimeStep: {time_step}")

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    env.set_fault_state(True)
    done = False

    steps = 0
    real_time = 0 / env.FRAMES_PER_SECOND

    flip = False
    #
    # states_data = pd.DataFrame(columns=["time_step", "x", "y", "z", "phi", "theta", "psi"])
    # states_data = states_data.append([steps, obs[0], obs[2], obs[4], obs[6], obs[8], obs[10]], ignore_index=True)

    if save_data:
        f = open(save_data_file_name, "w")

        states_data = pd.DataFrame(columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi"])

        new_df = pd.DataFrame([[steps, real_time, obs[0], obs[2], obs[4], obs[6], obs[8], obs[10]]],
                              columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi"])
        states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

    print(env.fault_map)
    while not done:
        action, _ = model.predict(obs)

        # if env.total_reward > 500:
        #     if not flip:
        #         env.handle_fault_injection()
        #         flip = True

        obs, reward, done, _ = env.step(action)

        print(
            '(%+0.2f,%+0.2f,%+0.2f) (%+0.2f,%+0.2f,%+0.2f)    steps = %04d    current_reward = %+0.2f    total_reward = %+0.2f' % (
            obs[0], obs[2], obs[4], obs[6], obs[8], obs[10], steps, reward, env.total_reward))
        # print(env.fault_map)
        env.render()

        sleep(1. / env.FRAMES_PER_SECOND)
        steps += 1
        real_time = steps / env.FRAMES_PER_SECOND

        if save_data:
            new_df = pd.DataFrame([[steps, real_time, obs[0] , obs[2], obs[4], obs[6], obs[8], obs[10]]],
                                  columns=["time_step", "real_time", "x", "y", "z", "phi", "theta", "psi"])
            states_data = pd.concat([states_data, new_df], axis=0, ignore_index=True)

            if steps > save_data_steps_limit:
                done = True

    print(env.total_reward)
    print(env.fault_map)
    env.close()

    if save_data:
        states_data.to_csv(save_data_file_name, index=False)


def main():
    episodes = 1

    fault_magnitude = [0.75, 1, 1, 1]

    for ep in range(episodes):
        env = gym.make("gym_copter:Hover3D-v29",
                       position_sigma=0.5, attitude_sigma=(np.pi / 5),
                       fault_magnitude=fault_magnitude)
        env.reset()

        viewer = ThreeDHoverRenderer(env,
                                     _heuristic,
                                     ())

        viewer.start()


main()
