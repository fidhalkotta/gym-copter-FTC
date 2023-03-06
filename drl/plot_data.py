import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

print("1")

print("2")

data = pd.read_csv("states_data.csv")

sns.set_theme()

figure, axis = plt.subplots(2, 3)

# Plot the responses for different events and regions
sns.lineplot(x="time_step", y="x", data=data, ax=axis[0, 0])
sns.lineplot(x=[0, 1200], y=[0, 0], color="g", ax=axis[0, 0])

sns.lineplot(x="time_step", y="y", data=data, ax=axis[0, 1])
sns.lineplot(x=[0, 1200], y=[0, 0], color="g", ax=axis[0, 1])

sns.lineplot(x="time_step", y="z", data=data, ax=axis[0, 2])
sns.lineplot(x=[0, 1200], y=[-5, -5], color="g", ax=axis[0, 2])
sns.lineplot(x=[500, 500], y=[-8, -0], color="r", ax=axis[0, 2])


sns.lineplot(x="time_step", y="phi", data=data, ax=axis[1, 0])
sns.lineplot(x=[0, 1200], y=[0, 0], color="g", ax=axis[1, 0])

sns.lineplot(x="time_step", y="theta", data=data, ax=axis[1, 1])
sns.lineplot(x=[0, 1200], y=[0, 0], color="g", ax=axis[1, 1])

sns.lineplot(x="time_step", y="psi", data=data, ax=axis[1, 2])

plt.suptitle("Position and Attitude with time with target values shown. 0.5 Fault at 500 Timestep")

# plt.savefig('save_as_a_png.png')
plt.show()

# plt.close(figure)

print("5")