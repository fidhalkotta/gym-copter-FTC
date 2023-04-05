import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

import scienceplots

print("1")

print("2")

# Get data
RL_data = pd.read_csv("data/gymCopter-Hover3DV28-fault-0_75-passive-p_sigma-0_5-a_sigma-0_2-1680339074-faulty.csv")
PID_data = pd.read_csv("data/gymCopter-Hover3DV28-fault-0_75-PID-faulty.csv")

# Data cleaning

# Multiply z to be positive upwards
RL_data.loc[:, "z"] *= -1
PID_data.loc[:, "z"] *= -1

# Convert angles to degrees
RL_data.loc[:, ["phi", "theta", "psi"]] *= 180/np.pi
PID_data.loc[:, ["phi", "theta", "psi"]] *= 180/np.pi

plt.style.use(['science', 'grid', 'high-vis'])

rows = 2
cols = 3

figure, axis = plt.subplots(rows, cols, sharex='row')

i = 0

ylims = [
    [-10, 10],
    [-10, 10],
    [-0.5, 10],
    [-90, 90],
    [-90, 90],
    [-90, 90],
]

ylabels = [
    "$x$ Position [m]",
    "$y$ Position [m]",
    "$z$ Position [m]",
    "$\phi$ Angle [degree]",
    "$\\theta$ Angle [degree]",
    "$\psi$ Angle [degree]",
]


for row in range(rows):
    for col in range(cols):
        RL_line = RL_data.iloc[:, [1, i+2]]\
            .plot(x="real_time",
                  ax=axis[row, col],
                  xlabel="Time [s]", ylabel=ylabels[i],
                  label="RL",
                  ylim=ylims[i]
              )
        PID_line = PID_data.iloc[:, [1, i+2]] \
            .plot(x="real_time",
                  ax=axis[row, col],
                  xlabel="Time [s]", ylabel=ylabels[i],
                  label="PID",
                  ylim=ylims[i]
              )
        final_point = PID_data.iloc[-1, [1, i+2]]

        axis[row, col].plot(final_point[0], final_point[1], 'x')

        i += 1


# RL_data.plot(x='real_time', y='x', ax=axis[0, 0], xlabel="Time [s]", ylabel="x Position [m]")
# RL_data.iloc[:, [1, 2]].plot(x="real_time", ax=axis[0, 1], xlabel="Time [s]", ylabel="x Position [m]")
# PID_data.plot(x='real_time', y='x', ax=axis[0, 0])

# s = pd.Series(data['x'])
# pd.plotting.autocorrelation_plot(s)


# # Plot the responses for different events and regions
# sns.lineplot(x=[0, 50], y=[0, 0], color="g", ax=axis[0, 0])
# sns.lineplot(x="real_time", y="x", data=data, ax=axis[0, 0])
# sns.lineplot(x="real_time", y="x", data=PID_data, ax=axis[0, 0])
#
# sns.lineplot(x=[0, 50], y=[0, 0], color="g", ax=axis[0, 1])
# sns.lineplot(x="real_time", y="y", data=data, ax=axis[0, 1])
# sns.lineplot(x="real_time", y="y", data=PID_data, ax=axis[0, 1])
#
# sns.lineplot(x=[0, 50], y=[-5, -5], color="g", ax=axis[0, 2])
# sns.lineplot(x="real_time", y="z", data=data, ax=axis[0, 2])
# sns.lineplot(x="real_time", y="z", data=PID_data, ax=axis[0, 2])
#
#
# sns.lineplot(x=[0, 50], y=[0, 0], color="g", ax=axis[1, 0])
# sns.lineplot(x="real_time", y="phi", data=data, ax=axis[1, 0])
# sns.lineplot(x="real_time", y="phi", data=PID_data, ax=axis[1, 0])
#
# sns.lineplot(x=[0, 50], y=[0, 0], color="g", ax=axis[1, 1])
# sns.lineplot(x="real_time", y="theta", data=data, ax=axis[1, 1])
# sns.lineplot(x="real_time", y="theta", data=PID_data, ax=axis[1, 1])
#
# sns.lineplot(x="real_time", y="psi", data=data, ax=axis[1, 2])
# sns.lineplot(x="real_time", y="psi", data=PID_data, ax=axis[1, 2])

plt.suptitle("Position and Attitude with time with target values shown. Hover3DV28 0.75 Fault")

# plt.savefig('save_as_a_png.png')
plt.show()

# plt.close(figure)

print("5")