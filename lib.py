import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def create_viz_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

class L1_average():
	def __init__(self, std_param, mode='linear'):
		self.std_param = std_param
		self.mode = mode

	def __call__(self, label, pred):
		return (np.sum(np.abs(pred - label)) / self.std_param)

	def jacobian(self, label, pred, expl_var):
		return np.array([
			np.sum((pred - label)* expl_var)*(1/self.std_param),
			np.sum((pred - label))*(1/self.std_param)
		])

class L2_average():
	def __init__(self, std_param, mode='linear'):
		self.std_param = std_param
		self.mode = mode


	def __call__(self, label, pred):
		return (np.sum((label - pred)**2) / self.std_param)

	def jacobian(self, label, pred, expl_var):
		return np.array([
			np.sum((label - pred)* expl_var)*(1/self.std_param)*(-2),


			np.sum((label - pred))*(1/self.std_param)*(-2)
		])


class LinearRegression():
	def __init__(self, data_path='./data.csv', lr=1e-3, nb_epochs=1000, a=0, b=0, viz=False, std=True):
		self.data_path = data_path
		self.datas = pd.read_csv(data_path)


		self.labels = np.array(self.datas['price'], dtype=float)
		self.explanatory_variable = np.array(self.datas['km'], dtype=float)

		self.std_label = np.std(self.labels)
		self.std_expl = np.std(self.explanatory_variable)

		self.mean_label = np.mean(self.labels)
		self.mean_expl = np.mean(self.explanatory_variable)

		if std:
			self.labels = self.zscore(self.labels, self.mean_label, self.std_label) # (self.labels - self.mean_label) / self.std_label
			self.explanatory_variable = self.zscore(self.explanatory_variable, self.mean_expl, self.std_expl) # (self.explanatory_variable - self.mean_expl) / self.std_expl

		self.lr = lr
		self.loss = L2_average(len(self.datas), 'linear')
		self.epochs = nb_epochs
		self.std = std
		self.a = a
		self.b = b

		self.loss_save = []
		self.slopes = []
		self.intercepts = []

		self.viz = viz


	def predict(self, x):
		return (self.a * x) + self.b

	def train(self):
		for epchs in range(self.epochs):
			preds = self.predict(self.explanatory_variable)
			loss = self.loss(self.labels, preds)
			jacobian = self.loss.jacobian(self.labels, preds, self.explanatory_variable)
			self.a -= jacobian[0] * self.lr
			self.b -= jacobian[1] * self.lr
			if self.viz and (epchs % 100 == 0):
				self.loss_save.append(loss)
				self.slopes.append(self.a)
				self.intercepts.append(self.b)
			if (epchs % 100 == 0):
				print(f'epochs {epchs + 1} loss -> ', loss)
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
