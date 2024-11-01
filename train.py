from lib import L2_average, LinearRegression, create_viz_folder
from visualizer import Visualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __name__ == '__main__':
	nb_epochs=20000
	model = LinearRegression(nb_epochs=nb_epochs, viz=True, std=True)

	ret = model.train()
	if len(sys.argv) == 2 and sys.argv[1] == 'viz':
		viz_folder = 'visualization/'
		create_viz_folder(viz_folder)

		x = 22899
		x2 = 139800

		x_norm = model.zscore(22899, model.mean_expl, model.std_expl) if model.std else x
		x2_norm = model.zscore(139800, model.mean_expl, model.std_expl) if model.std else x2

		# just to draw the line
		y = model.scale(model.predict(x_norm), model.mean_label, model.std_label) if model.std else model.predict(x)
		y2 = model.scale(model.predict(x2_norm), model.mean_label, model.std_label) if model.std else model.predict(x2)

		point1 = [x, y]
		point2 = [x2, y2]

		Z = pd.DataFrame({'norm_var': model.explanatory_variable, 'norm_label': model.labels})
		X = pd.read_csv('./data.csv')

		visualising = Visualizer(X)

		visualising.raw_data().line(point1, point2).to_file(viz_folder + 'proxy.html')
		visualising.loss_viz(model.loss_save, nb_epochs).to_file(viz_folder + 'loss_descent.html')

		visualising.df = Z if model.std else X
		loss = model.loss_save
		slope = model.slopes
		intercept = model.intercepts
		visualising.cost_function(model.loss, [slope, intercept, loss]).to_file(viz_folder + 'mse_surface_plot.html')

	model.save_weights('model.ckpt')
