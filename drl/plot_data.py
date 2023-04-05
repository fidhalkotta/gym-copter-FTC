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
RL_data.loc[:, ["phi", "theta", "psi"]] *= 180 / np.pi
PID_data.loc[:, ["phi", "theta", "psi"]] *= 180 / np.pi

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

symbols = [
    "x",
    "y",
    "z",
    "\phi",
    "\\theta",
    "\psi",
]


ylabels = [
    f"${symbols[0]}$ Position [m]",
    f"${symbols[1]}$ Position [m]",
    f"${symbols[2]}$ Position [m]",
    f"${symbols[3]}$ Angle [degree]",
    f"${symbols[4]}$ Angle [degree]",
    f"${symbols[5]}$ Angle [degree]",
]

reference_signals = [
    [[0, 50], [0, 0]],
    [[0, 50], [0, 0]],
    [[0, 0, 50], [8, 5, 5]],
    [[0, 50], [0, 0]],
    [[0, 50], [0, 0]],
    [],
]

for row in range(rows):
    for col in range(cols):
        if i != 5:
            axis[row, col].plot(reference_signals[i][0], reference_signals[i][1],
                                label="$" + symbols[i] + "_{ref}$")

        axis[row, col].plot(RL_data.iloc[:, 1], RL_data.iloc[:, i + 2],
                            label="$" + symbols[i] + "_{RL}$")
        axis[row, col].plot(PID_data.iloc[:, 1], PID_data.iloc[:, i + 2],
                            label="$" + symbols[i] + "_{PID}$")

        final_point = PID_data.iloc[-1, [1, i + 2]]
        axis[row, col].plot(final_point[0], final_point[1], 'x')

        axis[row, col].set_xlabel("Time [s]")
        axis[row, col].set_ylabel(ylabels[i])
        axis[row, col].set_ylim(ylims[i])
        axis[row, col].legend()

        i += 1

# for row in range(rows):
#     for col in range(cols):
#         RL_line = RL_data.iloc[:, [1, i+2]]\
#             .plot(x="real_time",
#                   ax=axis[row, col],
#                   xlabel="Time [s]", ylabel=ylabels[i],
#                   label="RL",
#                   ylim=ylims[i]
#               )
#         PID_line = PID_data.iloc[:, [1, i+2]] \
#             .plot(x="real_time",
#                   ax=axis[row, col],
#                   xlabel="Time [s]", ylabel=ylabels[i],
#                   label="PID",
#                   ylim=ylims[i]
#               )
#         final_point = PID_data.iloc[-1, [1, i+2]]
#
#         axis[row, col].plot(final_point[0], final_point[1], 'x')
#
#         i += 1


plt.suptitle("Position and Attitude with time with target values shown. Hover3DV28 0.75 Fault")

# plt.savefig('save_as_a_png.png')
plt.show()

# plt.close(figure)

print("5")
