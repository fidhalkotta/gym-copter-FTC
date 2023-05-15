
### Example Files (Useful for UCL markers)

The [folder](https://github.com/fidhalkotta/gym-copter-FTC/tree/master/example_files) `example_files` contains a few examples of the different aspects of the project. Including the following files:

- `ModelB.2.py` - This is the custom OpenAI Gym environment that I have created. It is based on the `gym-copter` environment, but I have added my own fault model and wind model.
- `trainModelB.2.py` - This is the file that trains the PPO model with StableBaselines3 integration, saving a model periodically.
- `loadModelB.2.py` - This is the file that loads the trained model and visualises the real time simulation, saving the data to a csv file.
- `plot_ModelB.2.ipynb` - This is a jupyter notebook that processes the data from the csv file and generates the graphs used in the report. An example of the graphs generated can be seen below.


<img src="images/faultyCaseComparison-v2.png" height="600">
