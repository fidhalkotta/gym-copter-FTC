import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

print("1")

print("2")

data = pd.read_csv("data/data_Hover3DV26-fault-0_75.csv")
PID_data = pd.read_csv("data/data_Hover3DV26-fault-0_75-PID.csv")

# sns.set_theme()
sns.set_style('white')
sns.set_context("paper", font_scale=2)

figure, axis = plt.subplots(2, 3)

# Plot the responses for different events and regions
sns.lineplot(x=[0, 20], y=[0, 0], color="g", ax=axis[0, 0])
sns.lineplot(x="real_time", y="x", data=data, ax=axis[0, 0])
sns.lineplot(x="real_time", y="x", data=PID_data, ax=axis[0, 0])

sns.lineplot(x=[0, 20], y=[0, 0], color="g", ax=axis[0, 1])
sns.lineplot(x="real_time", y="y", data=data, ax=axis[0, 1])
sns.lineplot(x="real_time", y="y", data=PID_data, ax=axis[0, 1])

sns.lineplot(x=[0, 20], y=[-5, -5], color="g", ax=axis[0, 2])
sns.lineplot(x="real_time", y="z", data=data, ax=axis[0, 2])
sns.lineplot(x="real_time", y="z", data=PID_data, ax=axis[0, 2])


sns.lineplot(x=[0, 20], y=[0, 0], color="g", ax=axis[1, 0])
sns.lineplot(x="real_time", y="phi", data=data, ax=axis[1, 0])
sns.lineplot(x="real_time", y="phi", data=PID_data, ax=axis[1, 0])

sns.lineplot(x=[0, 20], y=[0, 0], color="g", ax=axis[1, 1])
sns.lineplot(x="real_time", y="theta", data=data, ax=axis[1, 1])
sns.lineplot(x="real_time", y="theta", data=PID_data, ax=axis[1, 1])

sns.lineplot(x="real_time", y="psi", data=data, ax=axis[1, 2])
sns.lineplot(x="real_time", y="psi", data=PID_data, ax=axis[1, 2])

plt.suptitle("Position and Attitude with time with target values shown. Hover3DV26 0.75 Fault")

# plt.savefig('save_as_a_png.png')
plt.show()

# plt.close(figure)

print("5")