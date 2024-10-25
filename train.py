import numpy as np
import pandas as pd
import sys
from visualizer import Visualizer
import matplotlib.pyplot as plt
import random

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
	def __init__(self, data_path='./data.csv', lr=1e-5, nb_epochs=1000, precision_stop=0.0):
		self.data_path = data_path
		self.datas = pd.read_csv(data_path)

		self.labels = np.array(self.datas['price'], dtype=float)
		self.explanatory_variable = np.array(self.datas['km'], dtype=float)

		self.lr = lr
		self.precision_stop = precision_stop
		self.loss = L2_average(len(self.datas), 'linear')
		self.epochs = nb_epochs

		self.a = random.uniform(500, 1000)
		self.b = random.uniform(500, 1000)

		self.loss_save = []


	def predict(self, x):
		return (self.a * x) + self.b

	def update(self, params, jacobian):
		print(params, ' || ', (self.lr * jacobian))
		return (params - (self.lr * jacobian))

	def train(self):
		for epchs in range(self.epochs):
			preds = self.predict(self.explanatory_variable)
			loss = self.loss(self.labels, preds)
			self.loss_save.append(loss)
			jacobian = self.loss.jacobian(self.labels, preds, self.explanatory_variable)
			self.a -= jacobian[0] * self.lr
			self.b -= jacobian[1] * self.lr
			print(f'epochs {epchs} loss -> ', loss)
		return self.a, self.b

	def save_weight(self, filename):
		with open(filename, "w") as file:
			file.write(f"weights: [{self.a}, {self.b}]")





if __name__ == '__main__':
	test = LinearRegression(nb_epochs=1000000)

	ret = test.train()

	point1 = [1000, test.predict(1000)]
	point2 = [2000, test.predict(2000)]

	print('point1', point1)
	print('point2', point2)
	# print(ret)

	visualising = Visualizer(test.datas)

	visualising.raw_data().line(point1, point2).show()

	plt.plot(test.loss_save)
	plt.show()
	# print(test.loss_save)




