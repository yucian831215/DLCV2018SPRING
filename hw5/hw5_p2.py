import os
import pickle
import math
import keras
import csv
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras import backend as K

train_ground_truth_path = './HW5_data/TrimmedVideos/label/gt_train.csv'
valid_ground_truth_path = './HW5_data/TrimmedVideos/label/gt_valid.csv'
video_feature_path = './features'

def getVideoList(data_path):
    '''
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    '''
    result = {}

    with open (data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od

def LSTM_model():
	# Design LSTM model
	LSTM_inputs = Input(shape=(None, 2048))

	x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, name='lstm_1'), merge_mode='concat')(LSTM_inputs)
	x = LSTM(256, dropout=0.2, name='lstm_2')(x)

	x = Dense(128, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)

	x = Dense(11, activation='softmax', name='predictions')(x)

	model = Model(LSTM_inputs, x)

	return model

def pad(array):
	max_shape = max([feature.shape for feature in array])
	pad_array = []
	for feature in array:
		temp_array = np.zeros(max_shape)
		temp_array[:feature.shape[0], :feature.shape[1]] = feature
		pad_array.append(temp_array)

	return np.array(pad_array)

def load_data_batch(video_list, batch_size, data_size, state):
	# Shuffle train data 
	data_order = np.arange(1, data_size + 1)
	if state == 'train':
		np.random.shuffle(data_order)

	start_index = 0
	while True:
		X = []
		Y = []
		for index in range(batch_size):
			data_index = start_index + index
			path = os.path.join(video_feature_path, state)
			features = np.load(os.path.join(path, "Video_{}.npy".format(str(data_order[data_index]).zfill(4))))
			X.append(features)
			Y.append(int(video_list['Action_labels'][data_order[data_index] - 1]))

		X = np.array(X)
		X = pad(X)
		Y = np.array(Y).reshape(batch_size, 1)
		
		yield(X, Y)

		start_index += batch_size
		if start_index >= data_size:
			start_index = 0
			if state == 'train':
				np.random.shuffle(data_order)

if __name__ == '__main__':
	train_video_list = getVideoList(train_ground_truth_path)
	valid_video_list = getVideoList(valid_ground_truth_path)

	train_data_size = len(train_video_list['Video_index'])
	valid_data_size = len(valid_video_list['Video_index'])
	train_batch_size = 4
	valid_batch_size = 1
	train_batch_num = math.ceil(train_data_size / train_batch_size)
	valid_batch_num = math.ceil(valid_data_size / valid_batch_size)

	epochs = 150

	valid_generator = load_data_batch(valid_video_list, valid_batch_size, valid_data_size, "valid")
	train_generator = load_data_batch(train_video_list, train_batch_size, train_data_size, "train")

	lstm = LSTM_model()
	lstm.summary()

	lstm.compile(optimizer=Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	checkpoint = ModelCheckpoint('p2_lstm_model.h5', monitor='val_loss', save_best_only=True)

	history = lstm.fit_generator(train_generator,
					steps_per_epoch=train_batch_num, 
					epochs=epochs,   
					validation_data=valid_generator,
					validation_steps=valid_batch_num,
					callbacks=[checkpoint])


	# Record loss and accuracy
	curve_history = dict()
	curve_history['loss'] = history.history['loss']
	curve_history['accuracy'] = history.history['acc']
	curve_history['val_loss'] = history.history['val_loss']
	curve_history['val_accuracy'] = history.history['val_acc']
	
	with open('p2_curve_history.pickle', 'wb') as outfile:
		pickle.dump(curve_history, outfile)
