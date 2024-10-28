import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import L2_average

class Visualizer():
	def __init__(self, dataframe):
		self.df = dataframe
		self.col_names = self.df.columns
	def raw_data(self, prediction=None):
		'''
			this function is plotting the data as a cloud of point to visualize the relation between the variables
			if prediction is passed, it must be a numpy array of shape (x, 2) where x is the number of points you want to add
		'''
		plt.scatter(self.df[self.col_names[0]], self.df[self.col_names[1]])
		plt.xlabel(self.col_names[0])
		plt.ylabel(self.col_names[1])

		if prediction is not None:
			plt.scatter(prediction[0, 0], prediction[0, 1], color='red')

		return self

	def line(self, point1, point2):
		plt.axline(point1, point2)
		return self

	def cost_function(self, func, points=None):
		'''
			This function aims to plot our cost function and see the evolution / descent of our cost 
			regarding the update of the weights a and b (slope and intercept)

			func: the cost function defined like this: func(label, prediction) -> scalar
			points: [[slopes], [intercepts], [costs]] a list of points you want to highlights
		'''
		X = self.df.iloc[:, 0]
		Y = self.df.iloc[:, 1]
		# range of values for 
		w0_values = np.linspace(-1, 1, 100)
		w1_values = np.linspace(-1, 1, 100)
		
		W0, W1 = np.meshgrid(w0_values, w1_values)
		MSE_values = np.zeros_like(W0)

		for i in range(W0.shape[0]):
			for j in range(W0.shape[1]): 
				pred = W1[i, j] * X + W0[i, j]
				MSE_values[i, j] = func(Y, pred)

		fig = plt.figure(figsize=(10, 7))
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(W0, W1, MSE_values, cmap='viridis', edgecolor='none')
		ax.set_xlabel('w0 (Intercept)')
		ax.set_ylabel('w1 (Slope)')
		ax.set_zlabel('Loss')
		ax.set_title('3D plot of our cost function')
		if points is not None:
			w0 = points[1]
			w1 = points[0]
			loss = points[2]
			ax.scatter(w0, w1, loss, color='red', s=50, label="Training Points")
		plt.show()

	def show(self):
		plt.show()






if __name__ == '__main__':

	df = pd.read_csv('./data.csv')
	Z = df.apply(lambda col: (col - np.mean(col)) / np.std(col))

	X = Z['km']
	Y = Z['price']

	viz = Visualizer(Z)
	mean = L2_average(len(X))

	viz.cost_function(mean)