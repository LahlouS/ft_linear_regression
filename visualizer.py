import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import L2_average
import plotly.graph_objs as go
import plotly.io as pio


class Visualizer():
	def __init__(self, dataframe):
		self.df = dataframe
		self.col_names = self.df.columns
		self.fig = go.Figure()

	def raw_data(self, prediction=None):
		'''
		Plot the data as a cloud of points to visualize the relation between the variables.
		If `prediction` is provided, it should be a numpy array of shape (x, 2) where x is the number of points to add.
		'''
		# Scatter plot of the original data points
		scatter = go.Scatter(
			x=self.df[self.col_names[0]],
			y=self.df[self.col_names[1]],
			mode='markers',
			name='Data Points'
		)
		self.fig.add_trace(scatter)

		# If predictions are provided, add them as red points
		if prediction is not None:
			prediction_scatter = go.Scatter(
				x=prediction[:, 0],  # x-coordinates from prediction
				y=prediction[:, 1],  # y-coordinates from prediction
				mode='markers',
				marker=dict(color='red', size=5),
				name='Predictions'
			)
			self.fig.add_trace(prediction_scatter)

		# Update axis labels
		self.fig.update_layout(
			xaxis_title=self.col_names[0],
			yaxis_title=self.col_names[1]
		)

		return self

	def line(self, point1=None, point2=None):
		'''
		Plot a full line based on either:
		- slope and intercept, or
		- two points (point1 and point2), extending it to the full x-range.
		'''
		x_min, x_max = self.df[self.col_names[0]].min(), self.df[self.col_names[0]].max()

		if point1 is not None and point2 is not None:
			# Calculate slope and intercept from two points
			slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
			intercept = point1[1] - slope * point1[0]
			x_vals = np.array([x_min, x_max])
			y_vals = slope * x_vals + intercept

		else:
			raise ValueError("Provide either `slope` and `intercept` or `point1` and `point2`.")

		# Create the line plot trace
		line = go.Scatter(
			x=x_vals,
			y=y_vals,
			mode='lines',
			line=dict(color='blue'),
			name='Line'
		)
		self.fig.add_trace(line)

		return self

	def loss_viz(self, loss_serie, nepchs):

		x_vals = np.linspace(1, nepchs, int(nepchs / len(loss_serie)))

		line = go.Scatter(
			x=x_vals,
			y=loss_serie,
			mode='lines',
			line=dict(color='blue'),
			name='Line'
		)
		self.fig.add_trace(line)
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
		w0_values = np.linspace(-2, 1, 100)
		w1_values = np.linspace(-2, 1, 100)

		W0, W1 = np.meshgrid(w0_values, w1_values)
		MSE_values = np.zeros_like(W0)

		for i in range(W0.shape[0]):
			for j in range(W0.shape[1]):
				pred = W1[i, j] * X + W0[i, j]
				MSE_values[i, j] = func(Y, pred)

		surface = go.Surface(z=MSE_values, x=W0, y=W1, colorscale='Viridis', opacity=0.6)

		if points is not None:
			w0 = points[1]
			w1 = points[0]
			loss = points[2]
			scatter = go.Scatter3d(
								x=w0,
								y=w1,
								z=loss,
								mode='markers',
								marker=dict(size=1, color='red'),
								name='Training Points'
							)
		self.fig = go.Figure(data=[surface, scatter])

		# Customize layout
		self.fig.update_layout(
			title="3D Plot of MSE with Training Points",
			scene=dict(
				xaxis_title='w0 (Intercept)',
				yaxis_title='w1 (Slope)',
				zaxis_title='MSE'
			)
		)
		return self

	def to_file(self, filename):
		self.fig.update_layout(title="Interactive Plot")
		pio.write_html(self.fig, file=filename)  # Opens the plot in a web browser
		self.fig = go.Figure()







if __name__ == '__main__':

	df = pd.read_csv('./data.csv')
	Z = df.apply(lambda col: (col - np.mean(col)) / np.std(col))

	X = Z['km']
	Y = Z['price']

	viz = Visualizer(Z)
	mean = L2_average(len(X))

	viz.cost_function(mean)
