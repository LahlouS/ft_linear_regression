import numpy as np
import pandas as pd
import sys
from visualizer import Visualizer
import matplotlib.pyplot as plt
import random
import ast

class L2_average():
	def __init__(self, std_param, mode='linear'):
		self.std_param= std_param
		self.mode = mode


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

		self.labels = self.zscore(self.labels, self.mean_label, self.std_label) # (self.labels - self.mean_label) / self.std_label
		self.explanatory_variable = self.zscore(self.explanatory_variable, self.mean_expl, self.std_expl) # (self.explanatory_variable - self.mean_expl) / self.std_expl
		
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
		self.a = weights[0]
		self.b = weights[1]
		return weights

	def scale(self, x, mean, std):
		return x * std + mean

	def zscore(self, x, mean, std):
		return (x - mean) / std