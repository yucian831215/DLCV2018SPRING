import os
import json
import pickle
import math
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import generator as gen
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from keras.utils import plot_model
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import Adam, SGD
from keras.applications.inception_v3 import InceptionV3 as Inception

pathDataset = './dlcv_final_2_dataset'
pathInfo = './Facebank_info'
matchTable = dict()

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.EpochTrainLoss = []
		self.EpochTrainAcc = []
		self.EpochValidLoss = []
		self.EpochValidAcc = []

	def on_batch_end(self, batch, logs={}):
		self.loss.append(logs.get('loss', 0))

	def on_epoch_end(self, epoch, logs={}):
		self.EpochTrainLoss.append(logs.get('loss', 0))
		self.EpochTrainAcc.append(logs.get('acc', 0))
		self.EpochValidLoss.append(logs.get('val_loss', 0))
		self.EpochValidAcc.append(logs.get('val_acc', 0))

class mini_InceptionV3():
	def __init__(self, input_shape, classes):
		self.input_shape = input_shape
		self.classes = classes

	def depthwise_conv_block(self, x, filters, kernel_size, 
							 padding='same', strides=(1, 1)):
		separable_filters = math.ceil(filters / 2)

		x = SeparableConv2D(separable_filters, kernel_size, strides=strides,
							padding=padding, use_bias=False)(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)

		x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)

		return x

	def conv2d_bn(self, x, filters, kernel_size, drop=False,
		          padding='same', strides=(1, 1), name=None):

		if not name is None:
			bn_name = name + '_bn'
			conv_name = name + '_conv'
		else:
			bn_name = None
			conv_name = None

		x = Conv2D(filters, kernel_size,
				   strides=strides, padding=padding,
				   use_bias=False, name=conv_name)(x)
		if drop:
			x = Dropout(0.25)(x)
		x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
		x = Activation('relu', name=name)(x)

		return x

	def mini_InceptionV3_model(self):
		img_input = Input(shape=self.input_shape)

		x = self.conv2d_bn(img_input, 32, (3, 3), strides=(2, 2), padding='valid')
		x = self.conv2d_bn(x, 32, (3, 3), padding='valid')

		x = self.conv2d_bn(x, 64, (3, 3), strides=(2, 2), padding='valid')
		x = self.conv2d_bn(x, 64, (3, 3))

		x = self.conv2d_bn(x, 96, (1, 1), padding='valid')
		x = self.conv2d_bn(x, 128, (3, 3), strides=(2, 2), padding='valid')
		x = self.conv2d_bn(x, 128, (3, 3), padding='valid')

		branch1x1 = self.conv2d_bn(x, 96, (1, 1))
		branch5x5 = self.conv2d_bn(x, 48, (1, 1))
		branch5x5 = self.conv2d_bn(branch5x5, 64, (5, 5))
		branch3x3 = self.conv2d_bn(x, 64, (1, 1))
		branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3))
		branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3))
		branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
		branch_pool = self.conv2d_bn(branch_pool, 64, (1, 1))
		branches = [branch1x1, branch5x5, branch3x3, branch_pool]
		x = Concatenate(axis=3, name='mixed_0')(branches)

		branch_0 = self.conv2d_bn(x, 192, (3, 3), strides=(2, 2), padding='valid')
		branch_1 = self.conv2d_bn(x, 128, (1, 1))
		branch_1 = self.conv2d_bn(branch_1, 128, (3, 3))
		branch_1 = self.conv2d_bn(branch_1, 192, (3, 3), strides=(2, 2), padding='valid')
		branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
		branches = [branch_0, branch_1, branch_pool]
		x = Concatenate(axis=3, name='mixed_1')(branches)

		branch_0 = self.conv2d_bn(x, 128, (1, 1))
		branch_0 = self.conv2d_bn(branch_0, 192, (3, 3), strides=(2, 2), padding='valid')
		branch_1 = self.conv2d_bn(x, 128, (1, 1))
		branch_1 = self.conv2d_bn(branch_1, 144, (3, 3), strides=(2, 2), padding='valid')
		branch_2 = self.conv2d_bn(x, 128, (1, 1))
		branch_2 = self.conv2d_bn(branch_2, 144, (3, 3))
		branch_2 = self.conv2d_bn(branch_2, 160, (3, 3), strides=(2, 2), padding='valid')
		branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
		branches = [branch_0, branch_1, branch_2, branch_pool]
		x = Concatenate(axis=3, name='mixed_3')(branches)

		x = self.depthwise_conv_block(x, 1024, (3, 3))

		x = GlobalAveragePooling2D()(x)

		x = Dense(512, activation='relu', name='fc1')(x)
		x = Dropout(0.5)(x)

		x = Dense(self.classes, activation='softmax', name='preduction')(x)

		model = Model(img_input, x)

		return model

def load_matchTable():
	table = dict()
	pathTable = os.path.join(pathInfo, "matchTable.jsdb")
	with open(pathTable, 'r') as outfile:
		table = json.load(outfile)

	return table

def load_dataList(state):
	# dataList[name] = label
	dataList = list()
	labelList = list()
	path = os.path.join(pathDataset, "{}_id.txt".format(state))
	with open(path, 'r') as outfile:
		for outline in outfile:
			content = outline.split()
			dataList.append(os.path.join(pathDataset, state, content[0]))
			labelList.append(int(matchTable[content[1]]))

	return dataList, labelList

def SaveLoss(filename, history):
	if history:
		info_history = dict()
		info_history['BTloss'] = history.loss
		info_history['ETloss'] = history.EpochTrainLoss
		info_history['ETAcc'] = history.EpochTrainAcc
		info_history['EVloss'] = history.EpochValidLoss
		info_history['EVAcc'] = history.EpochValidAcc

		with open(filename, 'wb') as outfile:
			pickle.dump(info_history, outfile)

if __name__ == '__main__':
	matchTable = load_matchTable()

	model = mini_InceptionV3((218, 178, 3), 2360)
	mobile_model = model.mini_InceptionV3_model()
	plot_model(mobile_model, to_file='model.png')
	# mobile_model = load_model('./7726.h5')
	mobile_model.summary()

	trainDataList, trainLabelList = load_dataList('train')
	validDataList, validLabelList = load_dataList('val')

	epochs = 200
	train_data_size = len(trainDataList)
	valid_data_size = len(validDataList)
	train_batch_size = 16
	valid_batch_size = 1
	train_batch_num = int(math.ceil(train_data_size / train_batch_size))
	valid_batch_num = int(math.ceil(valid_data_size / valid_batch_size))

	train_generator = gen.dataGenerator(trainDataList, trainLabelList, batchSize=train_batch_size, shuffle=True, aug=True)
	valid_generator = gen.dataGenerator(validDataList, validLabelList, batchSize=valid_batch_size, shuffle=False, aug=False)

	mobile_model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	checkpoint = ModelCheckpoint('mini_inception_model.h5', monitor='val_loss', save_best_only=True)

	train_history = LossHistory()

	try:
		mobile_model.fit_generator(train_generator,
					steps_per_epoch=train_batch_num, 
					epochs=epochs,   
					validation_data=valid_generator,
					validation_steps=valid_batch_num,
					callbacks=[checkpoint, train_history])
		# Save train batch loss and info
		SaveLoss('InfoTrain.pickle', train_history)
	except KeyboardInterrupt:
		while True:
			Ask = "\nDo you want to save your training history? (y/n) : "
			check = input(Ask)
			if check in ['y', 'yes']:
				print('\n---- Save history ----\n\n')
				SaveLoss('InfoTrain.pickle', train_history)
				break
			elif check in ['n', 'no']:
				break
