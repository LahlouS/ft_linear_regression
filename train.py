import numpy as np
import pandas as pd
import sys
from visualizer import Visualizer
import matplotlib.pyplot as plt
import random
import ast

class L2_average():
	def __init__(self, std_param, model='linear'):
		self.std_param= std_param


	def __call__(self, label, pred):
		return (np.sum((label - pred)**2) / self.std_param)

	def jacobian(self, label, pred, expl_var):
		return np.array([
			np.sum((label - pred)* expl_var)*(1/self.std_param)*(-2),


			np.sum((label - pred))*(1/self.std_param)*(-2)
		])


class LinearRegression():
	def __init__(self, data_path='./data.csv', lr=1e-4, nb_epochs=1000, a=0, b=0):
		self.data_path = data_path
		self.datas = pd.read_csv(data_path)


		self.labels = np.array(self.datas['price'], dtype=float)
		self.explanatory_variable = np.array(self.datas['km'], dtype=float)

		self.std_label = np.std(self.labels)
		self.std_expl = np.std(self.explanatory_variable)
		
		self.mean_label = np.mean(self.labels)
		self.mean_expl = np.mean(self.explanatory_variable)

		self.labels = (self.labels - self.mean_label) / self.std_label
		self.explanatory_variable = (self.explanatory_variable - self.mean_expl) / self.std_expl
		
		self.lr = lr
		self.loss = L2_average(len(self.datas), 'linear')
		self.epochs = nb_epochs

		self.a = a
		self.b = b

		self.loss_save = []


	def predict(self, x):
		return (self.a * x) + self.b

	def train(self):
		for epchs in range(self.epochs):
			preds = self.predict(self.explanatory_variable)
			loss = self.loss(self.labels, preds)
			self.loss_save.append(loss)
			jacobian = self.loss.jacobian(self.labels, preds, self.explanatory_variable)
			self.a -= jacobian[0] * self.lr
			self.b -= jacobian[1] * self.lr
			# print(f'epochs {epchs + 1} loss -> ', loss)
		return self.a, self.b

	def save_weights(self, filename):
		with open(filename, "w") as file:
			file.write(f"weights: [{self.a}, {self.b}]")

	def read_weights(self, filename):
		with open(filename, 'r') as file:
			ret = file.readline()
		ret = ret[9:]
		weights = np.array(ast.literal_eval(ret))
		return weights


def scale(x, mean, std):
	return x * std + mean


if __name__ == '__main__':
	test = LinearRegression(nb_epochs=100000)

	ret = test.train()

	x = (240000 - test.mean_expl) / test.std_expl
	x2 = (139800 - test.mean_expl) / test.std_expl

	# just to draw the line
	point1 = [240000, scale(test.predict(x), test.mean_label, test.std_label)]
	point2 = [139800, scale(test.predict(x2), test.mean_label, test.std_label)]

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



