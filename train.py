from lib import L2_average, LinearRegression, create_viz_folder
from visualizer import Visualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __name__ == '__main__':
	nb_epochs=20000
	model = LinearRegression(nb_epochs=nb_epochs, viz=True)

	ret = model.train()
	if len(sys.argv) == 2 and sys.argv[1] == 'viz':
		viz_folder = 'visualization/'
		create_viz_folder(viz_folder)
		x = model.zscore(22899, model.mean_expl, model.std_expl)
		x2 = model.zscore(139800, model.mean_expl, model.std_expl)

		# just to draw the line
		point1 = [22899, model.scale(model.predict(x), model.mean_label, model.std_label)]
		point2 = [139800, model.scale(model.predict(x2), model.mean_label, model.std_label)]

		Z = pd.DataFrame({'norm_var': model.explanatory_variable, 'norm_label': model.labels})
		X = pd.read_csv('./data.csv')

		visualising = Visualizer(X)

		visualising.raw_data().line(point1, point2).to_file(viz_folder + 'proxy.html')
		visualising.loss_viz(model.loss_save, nb_epochs).to_file(viz_folder + 'loss_descent.html')

		visualising.df = Z
		loss = model.loss_save
		slope = model.slopes
		intercept = model.intercepts
		visualising.cost_function(model.loss, [slope, intercept, loss]).to_file(viz_folder + 'mse_surface_plot.html')

	model.save_weights('model.ckpt')
