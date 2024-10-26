import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

	def show(self):
		plt.show()

	

if __name__ == '__main__':
	datas = pd.read_csv('./data.csv', dtype=int)
	print(datas.head())
	visualising = Visualizer(datas)


	point = np.array([2,2])
	point2 = np.array([4,4])
	slope = 1

	visualising.raw_data().line(point, point2).show()

