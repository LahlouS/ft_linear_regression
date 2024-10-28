from lib import L2_average, LinearRegression
from visualizer import Visualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	model = LinearRegression(nb_epochs=10000, viz=True)

	ret = model.train()

	x = model.zscore(67000, model.mean_expl, model.std_expl)
	x2 = model.zscore(139800, model.mean_expl, model.std_expl)

	# just to draw the line
	point1 = [67000, model.scale(model.predict(x), model.mean_label, model.std_label)]
	point2 = [139800, model.scale(model.predict(x2), model.mean_label, model.std_label)]

	Z = pd.DataFrame({'norm_var': model.explanatory_variable, 'norm_label': model.labels})
	X = pd.read_csv('./data.csv')

	visualising = Visualizer(X)

	visualising.raw_data(np.array([point1, point2])).line(point1, point2).show()


	plt.plot(model.loss_save)
	plt.show()

	visualising.df = Z
	loss = model.loss_save
	slope = model.slopes
	intercept = model.intercepts
	visualising.cost_function(model.loss, [slope, intercept, loss])


	model.save_weights('model.ckpt')

	weights = model.read_weights('model.ckpt')



