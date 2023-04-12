import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

import scienceplots

# Get data
print("Getting Data")
RL_data = pd.read_csv("data/gymCopter-Hover3DV28-fault-0_75-passive-p_sigma-0_5-a_sigma-0_2-1680339074-faulty.csv")
PID_data = pd.read_csv("data/gymCopter-Hover3DV28-fault-0_75-PID-faulty.csv")

# Data cleaning
print("Cleaning Data")

# Multiply z to be positive upwards
RL_data.loc[:, "z"] *= -1
PID_data.loc[:, "z"] *= -1

# Convert angles to degrees
RL_data.loc[:, ["phi", "theta", "psi"]] *= 180 / np.pi
PID_data.loc[:, ["phi", "theta", "psi"]] *= 180 / np.pi




symbols = [
    "x",
    "y",
    "z",
    "\phi",
    "\\theta",
    "\psi",
]

ylims = [
    [-10, 10],
    [-10, 10],
    [-0.5, 10],
    [-90, 90],
    [-90, 90],
    [-90, 90],
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

plt.style.use(['science', 'grid', 'high-vis'])

rows = 2
cols = 3

figure, ax = plt.subplots(rows, cols, figsize=(13, 6.5),sharex='row')

i = 0

print("Plotting Data")
for row in range(rows):
    for col in range(cols):
        if i != 5:
            ax[row, col].plot(reference_signals[i][0], reference_signals[i][1],
                              label="$" + symbols[i] + "_{ref}$")

        next(ax[row, col]._get_lines.prop_cycler) if i == 5 else None

        ax[row, col].plot(RL_data.iloc[:, 1], RL_data.iloc[:, i + 2],
                          label="$" + symbols[i] + "_{RL}$")
        ax[row, col].plot(PID_data.iloc[:, 1], PID_data.iloc[:, i + 2],
                          label="$" + symbols[i] + "_{PID}$")

        final_point = PID_data.iloc[-1, [1, i + 2]]
        ax[row, col].plot(final_point[0], final_point[1], 'x')

        ax[row, col].set_xlabel("Time [s]", fontsize=8) if i >= 3 else None
        ax[row, col].set_ylabel(ylabels[i], fontsize=8)
        ax[row, col].set_ylim(ylims[i])
        ax[row, col].tick_params(axis='both', which='major', labelsize=8)
        ax[row, col].legend(fontsize=8, fancybox=False, edgecolor='black')

        i += 1

plt.suptitle("Position and Attitude Response of Model $A$ with $m_1 = 0.75$ and $m_2 = m_3 = m_4 = 1$")

# plt.savefig('images/Hover3DV28-fault-0_75-passive-faulty.png', dpi=300)
plt.show()

print("Closing Plots")
plt.close(figure)
