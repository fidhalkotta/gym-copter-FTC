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

# 3D hover V5: Gaussian reward based on distance y offset
register(
    id='Hover3D-v5',
    entry_point='gym_copter.envs:Hover3DV5',
    max_episode_steps=1000
)

# 3D hover V6: Just targeting z plane, increase time steps to 20,000 for Training
register(
    id='Hover3D-v6',
    entry_point='gym_copter.envs:Hover3DV6',
    max_episode_steps=20000
)

# 3D hover V7: Simple + 1 per frame alive. No Pitch or Roll Bounds
register(
    id='Hover3D-v7',
    entry_point='gym_copter.envs:Hover3DV7',
    max_episode_steps=20000
)

# 3D hover V8: More priority for z plane, some reward for x and y
register(
    id='Hover3D-v8',
    entry_point='gym_copter.envs:Hover3DV8',
    max_episode_steps=20000
)

# 3D hover V9: More priority for z plane, some reward for x and y, and phi and theta
register(
    id='Hover3D-v9',
    entry_point='gym_copter.envs:Hover3DV9',
    max_episode_steps=20000
)

# 3D hover V10: Just z plane no Gaussian
register(
    id='Hover3D-v10',
    entry_point='gym_copter.envs:Hover3DV10',
    max_episode_steps=20000
)

# 3D hover V12: r_a from Jiang and Lynch
register(
    id='Hover3D-v12',
    entry_point='gym_copter.envs:Hover3DV12',
    max_episode_steps=20000
)

# 3D hover V13: r_a but more positive from Jiang and Lynch
register(
    id='Hover3D-v13',
    entry_point='gym_copter.envs:Hover3DV13',
    max_episode_steps=20000
)

# bug fix 23/02/23

# 3D hover V14: gaussian x,y,z episode doesnt terminate
register(
    id='Hover3D-v14',
    entry_point='gym_copter.envs:Hover3DV14',
    max_episode_steps=20000
)

# 3D hover V15: gaussian x,y,z but episode terminates on out of bounds
register(
    id='Hover3D-v15',
    entry_point='gym_copter.envs:Hover3DV15',
    max_episode_steps=20000
)

# 3D hover V16: gaussian x,y,z episode doesnt terminate but no random initial position
register(
    id='Hover3D-v16',
    entry_point='gym_copter.envs:Hover3DV16',
    max_episode_steps=20000
)

# 3D hover V17: gaussian x,y,z episode does terminate but no random initial position
register(
    id='Hover3D-v17',
    entry_point='gym_copter.envs:Hover3DV17',
    max_episode_steps=20000
)

# 3D hover V18: gaussian x,y,z, phi, theta episode does terminate but no random initial position
register(
    id='Hover3D-v18',
    entry_point='gym_copter.envs:Hover3DV18',
    max_episode_steps=20000
)

# 3D hover V19: Similar to V10, no random init position, no termination
register(
    id='Hover3D-v19',
    entry_point='gym_copter.envs:Hover3DV19',
    max_episode_steps=20000
)

# 3D hover V20: same as V19 but limiting angles to 2pi
register(
    id='Hover3D-v20',
    entry_point='gym_copter.envs:Hover3DV20',
    max_episode_steps=20000
)

# 3D hover V21: same as V18 Fault half the time
register(
    id='Hover3D-v21',
    entry_point='gym_copter.envs:Hover3DV21',
    max_episode_steps=20000
)

# 3D hover V22: same as V21 but yaw also in reward
register(
    id='Hover3D-v22',
    entry_point='gym_copter.envs:Hover3DV22',
    max_episode_steps=20000
)

# 3D hover V23: same as V22 but always faulty
register(
    id='Hover3D-v23',
    entry_point='gym_copter.envs:Hover3DV23',
    max_episode_steps=20000
)

# 3D hover V24: same as V23, always faulty, but now with angular rates instead of attitude dot, body frame rates instead of inertial frame
register(
    id='Hover3D-v24',
    entry_point='gym_copter.envs:Hover3DV24',
    max_episode_steps=20000
)

# 3D hover V25: same as V23, always faulty with psidot instead of psu
register(
    id='Hover3D-v25',
    entry_point='gym_copter.envs:Hover3DV25',
    max_episode_steps=20000
)

# Bug fix 26/03/23 No longer using incorrect bodyToInertial in dynamics

# 3D hover V26: same as V24, Fixed bodyToInertial funciton in dynamics
register(
    id='Hover3D-v26',
    entry_point='gym_copter.envs:Hover3DV26',
    max_episode_steps=20000
)

# 3D hover V27: Same as V26, instead of all attitude rates, theta and phi are just values and psi is psirate
register(
    id='Hover3D-v27',
    entry_point='gym_copter.envs:Hover3DV27',
    max_episode_steps=20000
)

# 3D hover V28: Same as V26, but variable sigmas
register(
    id='Hover3D-v28',
    entry_point='gym_copter.envs:Hover3DV28',
    max_episode_steps=20000
)

# 3D hover V29: Same as V28, but wind enabled
register(
    id='Hover3D-v29',
    entry_point='gym_copter.envs:Hover3DV29',
    max_episode_steps=20000
)

# 3D hover V23: Same as V29 but terminate on out of bounds and lose bigely
register(
    id='Hover3D-v30',
    entry_point='gym_copter.envs:Hover3DV30',
    max_episode_steps=20000
)

# Final Experimentation - All have wind and no out of bounds termination

# Model A V1: Negative Squared Error
register(
    id='ModelA-v1',
    entry_point='gym_copter.envs:ModelA_V1',
    max_episode_steps=20_000
)

# Model A V2: Negative Squared Error with termination on bounds
register(
    id='ModelA-v2',
    entry_point='gym_copter.envs:ModelA_V2',
    max_episode_steps=20_000
)

# Model B V1: Gaussian Distribution
register(
    id='ModelB-v1',
    entry_point='gym_copter.envs:ModelB_V1',
    max_episode_steps=20_000
)

# Model B V2: Passive Fault Training
register(
    id='ModelB-v2',
    entry_point='gym_copter.envs:ModelB_V2',
    max_episode_steps=20_000
)

# Model B V3: Passive Fault Training - with multiple fault cases
register(
    id='ModelB-v3',
    entry_point='gym_copter.envs:ModelB_V3',
    max_episode_steps=20_000
)



print("Registered Envs")