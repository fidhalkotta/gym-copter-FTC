import gym as gym
from stable_baselines3 import PPO, DDPG
from time import sleep

project_name = "gymCopter-Lander2D-PPO-1673024145"
models_dir = f"models/{project_name}"

env = gym.make("gym_copter:Lander2D-v0")

initialPose = (-1, -1, 10, 0, 0)

env.reset()

model_path = f"{models_dir}/280000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    print(f"Trying ep: {ep}")
    while not done:
        action, _states = model.predict(obs)
        state, reward, done, _ = env.step(action)
        env.render()

        sleep(1. / env.FRAMES_PER_SECOND)
        # print(rewards)