from lib import L2_average, LinearRegression
from visualizer import Visualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	test = LinearRegression(nb_epochs=100000)

	ret = test.train()

	x = test.zscore(67000, test.mean_expl, test.std_expl)
	x2 = test.zscore(139800, test.mean_expl, test.std_expl)

	# just to draw the line
	point1 = [67000, test.scale(test.predict(x), test.mean_label, test.std_label)]
	point2 = [139800, test.scale(test.predict(x2), test.mean_label, test.std_label)]

	# on the original scale
	print('test.std_expl ->', test.std_expl)
	print('test.std_label ->', test.std_label)
	print('test.mean_expl ->', test.mean_expl)
	print('test.mean_label ->', test.mean_label)

	# working with the normalised variable since model is trained on the none normalised variables
	Z = pd.DataFrame({'norm_var': test.explanatory_variable, 'norm_label': test.labels})
	X = pd.read_csv('./data.csv')

	visualising = Visualizer(X)

	print('point1 --> ', point1)
	print('point2 --> ', point2)
	visualising.raw_data(np.array([point1, point2])).line(point1, point2).show()

	plt.plot(test.loss_save)
	plt.show()


	test.save_weights('model.ckpt')

	weights = test.read_weights('model.ckpt')



