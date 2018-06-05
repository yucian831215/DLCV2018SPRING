import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline

graph_data_path = './p1_curve_history.pickle'

def loss_graph():
	with open(graph_data_path, 'rb') as outfile:
		curve_history = pickle.load(outfile)
	plt.figure(figsize=(16,6))

	# train curve
	ax1 = plt.subplot(1, 2, 1)
	loss_history_X = np.arange(len(curve_history['loss']))
	loss_history_Y = np.array(curve_history['loss'])
	smooth = InterpolatedUnivariateSpline(loss_history_X, loss_history_Y)
	loss_history_X_smooth = np.linspace(loss_history_X.min(), loss_history_X.max(), num=500).astype(np.int32)
	loss_history_Y_smooth = smooth(loss_history_X_smooth)

	accuracy_history_X = np.arange(len(curve_history['accuracy']))
	accuracy_history_Y = np.array(curve_history['accuracy'])
	smooth = UnivariateSpline(accuracy_history_X, accuracy_history_Y)
	accuracy_history_X_smooth = np.linspace(accuracy_history_X.min(), accuracy_history_X.max(), num=500).astype(np.int32)
	accuracy_history_Y_smooth = smooth(accuracy_history_X_smooth)

	ax1.plot(loss_history_X, loss_history_Y, color='mistyrose', linewidth=1)

	train_loss, = ax1.plot(loss_history_X_smooth, loss_history_Y_smooth, color='r', linewidth=1.5, label='train_loss')

	ax2 = ax1.twinx()
	ax2.set_ylim(0.2, 0.8)
	ax2.set_yticks(np.arange(0.2, 0.8, 0.1))
	train_acc, = ax2.plot(accuracy_history_X_smooth, accuracy_history_Y_smooth, color='b', linewidth=1.5, label='train_acc')

	plt.legend(handles=[train_loss, train_acc])
	ax1.set_xlabel("Training Steps")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")
	plt.title("Train performance")

	# valid curve
	ax1 = plt.subplot(1, 2, 2)
	valid_loss, = ax1.plot(curve_history['val_loss'], color='r', linewidth=1.5, label='valid_loss')

	ax2 = ax1.twinx()
	ax2.set_ylim(0.2, 0.8)
	ax2.set_yticks(np.arange(0.2, 0.8, 0.1))
	valid_acc, = ax2.plot(curve_history['val_accuracy'], color='b', linewidth=1.5, label='valid_acc')
	baseline, = ax2.plot(0.45 * np.ones(100), color='g', linewidth=1.5, label='baseline')

	plt.legend(handles=[valid_loss, valid_acc, baseline])
	ax1.set_xlabel("Training Epochs")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")
	plt.title("Validation performance")	

	plt.savefig("fig1_2.jpg")
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='input path', type=str)
	args = parser.parse_args()

	loss_graph()