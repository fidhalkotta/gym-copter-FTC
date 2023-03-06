import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so


class Plotter:
    def __init__(self):
        # sns.set_theme(style="darkgrid")
        self.plot_fig = None

    def plot(self, states, steps):

        # Load an example dataset with long-form data
        self.plot_fig = plt.Figure()

        xs = []

        for s in states:
            xs.append(s[0])

        # Plot the responses for different events and regions
        sns.lineplot(x=range(len(states)), y=xs)

        plt.savefig('save_as_a_png.png')

        plt.close(self.plot_fig)

