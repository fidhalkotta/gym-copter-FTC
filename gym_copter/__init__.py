'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

# 1D lander
register(
    id='Lander1D-v0',
    entry_point='gym_copter.envs:Lander1D',
    max_episode_steps=400
)

# 1D hover
register(
    id='Hover1D-v0',
    entry_point='gym_copter.envs:Hover1D',
    max_episode_steps=1000
)

# 2D lander
register(
    id='Lander2D-v0',
    entry_point='gym_copter.envs:Lander2D',
    max_episode_steps=400
)

# 2D hover
register(
    id='Hover2D-v0',
    entry_point='gym_copter.envs:Hover2D',
    max_episode_steps=1000
)

# 3D lander
register(
    id='Lander3D-v0',
    entry_point='gym_copter.envs:Lander3D',
    max_episode_steps=400
)

# 3D hover
register(
    id='Hover3D-v0',
    entry_point='gym_copter.envs:Hover3D',
    max_episode_steps=1000
)

###################

# 3D hover V1: Reward from lander reward, it is change in position sort of. Also incorrect target
register(
    id='Hover3D-v1',
    entry_point='gym_copter.envs:Hover3DV1',
    max_episode_steps=1000
)

# 3D hover V2: Distance from target
register(
    id='Hover3D-v2',
    entry_point='gym_copter.envs:Hover3DV2',
    max_episode_steps=1000
)

# 3D hover V3: Gaussian reward based on distance.
register(
    id='Hover3D-v3',
    entry_point='gym_copter.envs:Hover3DV3',
    max_episode_steps=1000
)

# 3D hover V4: Gaussian reward based on distance.
register(
    id='Hover3D-v4',
    entry_point='gym_copter.envs:Hover3DV4',
    max_episode_steps=1000
)

# 3D hover V5: Gaussian reward based on distance.
register(
    id='Hover3D-v5',
    entry_point='gym_copter.envs:Hover3DV5',
    max_episode_steps=1000
)



print("Registered Envs")