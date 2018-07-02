import os
import pickle
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline

def curve_smooth(scalers, weight):
	last = scalers[0]
	smoothed = list()
	for point in scalers:
		smoothed_val = last * weight + (1 - weight) * point
		smoothed.append(smoothed_val)
		last = smoothed_val
	return np.array(smoothed)

def find_CombineIndex(arr1, arr2):
	index = 0
	for arr_index in range(len(arr1)):
		if arr1[arr_index] <= arr2[0]:
			index = arr_index - 1
			break

	return index

def loss_graph():
	with open('./InfoTrain_aug_1.pickle', 'rb') as outfile:
		aug_history_1 = pickle.load(outfile)

	with open('./InfoTrain_aug_2.pickle', 'rb') as outfile:
		aug_history_2 = pickle.load(outfile)

	with open('./InfoTrain_aug_3.pickle', 'rb') as outfile:
		aug_history_3 = pickle.load(outfile)

	# Epochs 1 ~ 100
	BTloss_1 = np.array(aug_history_1['BTloss'])
	ETloss_1 = np.array(aug_history_1['ETloss'])
	ETAcc_1 = np.array(aug_history_1['ETAcc'])
	EVloss_1 = np.array(aug_history_1['EVloss'])
	EVAcc_1 = np.array(aug_history_1['EVAcc'])
	# Epochs 101 ~ 200
	BTloss_2 = np.array(aug_history_2['BTloss'])
	ETloss_2 = np.array(aug_history_2['ETloss'])
	ETAcc_2 = np.array(aug_history_2['ETAcc'])
	EVloss_2 = np.array(aug_history_2['EVloss'])
	EVAcc_2 = np.array(aug_history_2['EVAcc'])
	# Epochs 201 ~ 300
	BTloss_3 = np.array(aug_history_3['BTloss'])
	ETloss_3 = np.array(aug_history_3['ETloss'])
	ETAcc_3 = np.array(aug_history_3['ETAcc'])
	EVloss_3 = np.array(aug_history_3['EVloss'])
	EVAcc_3 = np.array(aug_history_3['EVAcc'])

	CombineIndex = find_CombineIndex(ETloss_1, ETloss_2)

	BTloss = np.concatenate([BTloss_1[0 : CombineIndex*3530], BTloss_2[0 : 100*3530]])
	ETloss = np.concatenate([ETloss_1[0 : CombineIndex], ETloss_2[0:]])
	ETAcc = np.concatenate([ETAcc_1[0 : CombineIndex], ETAcc_2[0:]])
	EVloss = np.concatenate([EVloss_1[0 : CombineIndex], EVloss_2[0:]])
	EVAcc = np.concatenate([EVAcc_1[0 : CombineIndex], EVAcc_2[0:]])

	CombineIndex = find_CombineIndex(ETloss, ETloss_3)

	num = len(ETloss_3) - 12 
	BTloss = np.concatenate([BTloss[0 : CombineIndex*3530], BTloss_3[0 : num*3530]])
	ETloss = np.concatenate([ETloss[0 : CombineIndex], ETloss_3[0 : num]])
	ETAcc = np.concatenate([ETAcc[0 : CombineIndex], ETAcc_3[0 : num]])
	EVloss = np.concatenate([EVloss[0 : CombineIndex], EVloss_3[0 : num]])
	EVAcc = np.concatenate([EVAcc[0 : CombineIndex], EVAcc_3[0 : num]])

    # With aug
	plt.figure(figsize=(16,6))
    # train curve
	ax1 = plt.subplot(1, 2, 1)
	ax2 = ax1.twinx()
	
	loss_history_X = np.arange(len(BTloss))
	loss_history_Y = BTloss
	smooth = InterpolatedUnivariateSpline(loss_history_X, loss_history_Y)
	loss_history_X_smooth = np.linspace(loss_history_X.min(), loss_history_X.max(), num=200).astype(np.int32)
	loss_history_Y_smooth = smooth(loss_history_X_smooth)
	loss_history_Y_smooth = curve_smooth(loss_history_Y_smooth, 0.6)

	plot_X = [index for index in range(len(loss_history_X)) if index % 50 == 0]
	plot_Y = [loss_history_Y[index] for index in range(len(loss_history_Y)) if index % 50 == 0]
	plot_Y = curve_smooth(plot_Y, 0.6)

	ax1.plot(np.array(plot_X) / 3530, plot_Y, color='mistyrose', linewidth=1)
	train_loss, = ax1.plot(loss_history_X_smooth / 3530, loss_history_Y_smooth, color='r', linewidth=1.5, label='train_loss')

	accuracy_history_X = np.arange(len(ETAcc))
	accuracy_history_Y = ETAcc
	accuracy_history_Y = curve_smooth(accuracy_history_Y, 0.6)

	train_acc, = ax2.plot(accuracy_history_X, accuracy_history_Y, color='b', linewidth=1.5, label='train_acc')

	plt.legend(handles=[train_loss, train_acc])
	ax1.set_xlim(0, len(loss_history_X) / 3530)
	ax1.set_ylim(0, )
	ax2.set_ylim(0, 1.0)
	ax1.set_xlabel("Training Epoch")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")

	# valid curve
	ax1 = plt.subplot(1, 2, 2)
	ax2 = ax1.twinx()

	loss_history_X = np.arange(len(EVloss))
	loss_history_Y = EVloss
	loss_history_Y = curve_smooth(loss_history_Y, 0.6)

	valid_loss, = ax1.plot(loss_history_X, loss_history_Y, color='r', linewidth=1.5, label='valid_loss')
	
	accuracy_history_X = np.arange(len(EVAcc))
	accuracy_history_Y = EVAcc
	accuracy_history_Y = curve_smooth(accuracy_history_Y, 0.6)

	valid_acc, = ax2.plot(accuracy_history_X, accuracy_history_Y, color='b', linewidth=1.5, label='valid_acc')

	plt.legend(handles=[valid_loss, valid_acc], loc=3)
	ax1.set_xlim(0, len(loss_history_X))
	ax2.set_ylim(0, )
	ax1.set_xlabel("Training Epoch")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")
	
	plt.savefig("augment.jpg")
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='input path', type=str)
	args = parser.parse_args()

	loss_graph()
