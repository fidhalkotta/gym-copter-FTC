from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import tensorboard as tb

import scienceplots
#%%
major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)
#%%
experiment_id = "s3uzotOOQoCBkGQxtPbELg"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

# Select only the rows where the "tag" column has the values "rollout/ep_len_mean" or "rollout/ep_rew_mean"
df = df[df['tag'].isin(['rollout/ep_len_mean', 'rollout/ep_rew_mean'])]
df = tb.data.experimental.utils.pivot_dataframe(df)

# Rename the columns to "ep_len_mean" and "ep_rew_mean"
df = df.rename(columns={'rollout/ep_len_mean': 'ep_len_mean', 'rollout/ep_rew_mean': 'ep_rew_mean'})

df.info()
#%%
# plt.style.use(['science', 'grid', 'high-vis'])
#
# fig = sns.lineplot(data=df, x="step", y="ep_len_mean").set_title("accuracy")
# plt.show()
#%%
plt.style.use(['science', 'grid', 'high-vis'])
plt.plot(df["step"], df["ep_len_mean"])
plt.show()