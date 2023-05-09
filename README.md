<img src="media/RL-ModelB.2-faulty.gif" height="300">

# Fault Tolerant Quadcopter Control using Reinforcement Learning



OpenAI Gym environment for fault tolerant control of a quadcopter using Stable-Baselines3, extended on top of [gym-copter](https://github.com/simondlevy/gym-copter). Repository includes data processing and graphic generation. 
 
## UCL MECH0020: Individual Project

This repository contains all of the code for my 3rd Year MECH0020 project. This was built on top of an existing simulation environment made by Simon D. Levy called [gym-copter](https://github.com/simondlevy/gym-copter). 

I have extended the environment, by developing my own fault model, which allows for a loss of effectiveness to be simulated. I have integrated Stable-Baselines3 into this environment and used the reinforcement learning algorithm PPO to train the model to control the quadcopter in both the nominal case (no faults) and the faulty case (loss of effectiveness in one motor).

I have also integrated a continuous wind disturbance model that allows for a more realistic and complex simulation environment.




## Future Plans

### Immediate Future

- [ ] Clean up code and make it more readable
- [ ] Make repository more open-source friendly
- [ ] Clean and optimise the data processing in jupyter notebooks

### Long Term Future
- [ ] Make the code more modular
- [ ] Add more fault models
- [ ] Add more reinforcement learning algorithms
- [ ] Add more quadcopter models
- [ ] Add more wind models
