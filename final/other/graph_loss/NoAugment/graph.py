import os
import pickle
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline

graph_data_path = './history_ori.pickle'

def curve_smooth(scalers, weight):
	last = scalers[0]
	smoothed = list()
	for point in scalers:
		smoothed_val = last * weight + (1 - weight) * point
		smoothed.append(smoothed_val)
		last = smoothed_val
	return np.array(smoothed)

def loss_graph():
	with open('./InfoTrain_ori.pickle', 'rb') as outfile:
		ori_history = pickle.load(outfile)
        # Without aug
	plt.figure(figsize=(16,6))
        # train curve
	ax1 = plt.subplot(1, 2, 1)
	ax2 = ax1.twinx()
	
	loss_history_X = np.arange(len(ori_history['BTloss']))
	loss_history_Y = np.array(ori_history['BTloss'])
	smooth = InterpolatedUnivariateSpline(loss_history_X, loss_history_Y)
	loss_history_X_smooth = np.linspace(loss_history_X.min(), loss_history_X.max(), num=200).astype(np.int32)
	loss_history_Y_smooth = smooth(loss_history_X_smooth)
	loss_history_Y_smooth = curve_smooth(loss_history_Y_smooth, 0.6)

	plot_X = [index for index in range(len(loss_history_X)) if index % 50 == 0]
	plot_Y = [loss_history_Y[index] for index in range(len(loss_history_Y)) if index % 50 == 0]
	plot_Y = curve_smooth(plot_Y, 0.6)

	ax1.plot(np.array(plot_X) / 3530, plot_Y, color='mistyrose', linewidth=1)
	train_loss, = ax1.plot(loss_history_X_smooth / 3530, loss_history_Y_smooth, color='r', linewidth=1.5, label='train_loss')

	accuracy_history_X = np.arange(len(ori_history['ETAcc']))
	accuracy_history_Y = np.array(ori_history['ETAcc'])
	accuracy_history_Y = curve_smooth(accuracy_history_Y, 0.6)

	train_acc, = ax2.plot(accuracy_history_X, accuracy_history_Y, color='b', linewidth=1.5, label='train_acc')

	plt.legend(handles=[train_loss, train_acc])
	ax1.set_xlim(0, len(loss_history_X) / 3530)
	ax1.set_ylim(0, 8)
	ax2.set_ylim(0, 1.0)
	ax1.set_xlabel("Training Epoch")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")

	# valid curve
	ax1 = plt.subplot(1, 2, 2)
	ax2 = ax1.twinx()

	loss_history_X = np.arange(len(ori_history['EVloss']))
	loss_history_Y = np.array(ori_history['EVloss'])
	loss_history_Y = curve_smooth(loss_history_Y, 0.6)

	valid_loss, = ax1.plot(loss_history_X, loss_history_Y, color='r', linewidth=1.5, label='valid_loss')
	
	accuracy_history_X = np.arange(len(ori_history['EVAcc']))
	accuracy_history_Y = np.array(ori_history['EVAcc'])
	accuracy_history_Y = curve_smooth(accuracy_history_Y, 0.6)

	valid_acc, = ax2.plot(accuracy_history_X, accuracy_history_Y, color='b', linewidth=1.5, label='valid_acc')

	plt.legend(handles=[valid_loss, valid_acc], loc=4)
	ax1.set_xlim(0, len(loss_history_X))
	ax2.set_ylim(0, 0.4)
	ax1.set_xlabel("Training Epoch")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")
	
	plt.savefig("original.jpg")
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='input path', type=str)
	args = parser.parse_args()

	loss_graph()
