import os
import pickle
import math
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from skimage import io

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3 as Inception

video_path = './HW5_data/FullLengthVideos/videos'
label_path = './HW5_data/FullLengthVideos/labels'
inception_model_path = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
video_feature_path = './full_features'
lstm_weights_path = './model/p2_lstm_model.h5'

def getVideoList(data_path):
	video_list = [video_name for video_name in os.listdir(data_path)]
	video_list.sort()

	return video_list

def normalize_image(img):
	output_img = img / 255
	return output_img

def Inception_extract_features_model():
	# Build conv and pooling layers according to inception
	model = Inception(include_top=False, weights=None, input_shape=(240, 320, 3))
	model.load_weights(inception_model_path)
	Inception_inputs = Input(shape=(240, 320, 3))
	Inception_outputs = model(Inception_inputs)
	model = Model(Inception_inputs, GlobalAveragePooling2D()(Inception_outputs))

	return model

def LSTM_model():
	# Design LSTM model
	LSTM_inputs = Input(shape=(None, 2048))

	x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, name='lstm_1'), merge_mode='concat')(LSTM_inputs)
	x = LSTM(256, return_sequences=True, dropout=0.2, name='lstm_2')(x)

	x = TimeDistributed(Dense(128, activation='relu', name='fc1'))(x)
	x = TimeDistributed(Dropout(0.5))(x)

	x = TimeDistributed(Dense(11, activation='softmax', name='predictions'))(x)

	model = Model(LSTM_inputs, x)

	return model

def Extract_features(model, video_list, state):
	path = os.path.join(video_path, state)
	save_feature_path = os.path.join(video_feature_path, state)

	for video_name in video_list:
		video_image_path = os.path.join(path, video_name)
		video_image_list = getVideoList(video_image_path)
		image_feature = []
		# Extract image feature by Inception model
		for image in video_image_list:
			image = io.imread(os.path.join(video_image_path, image))
			image = normalize_image(image).reshape((1, ) + image.shape)
			image_feature.append(model.predict(image))
			print("{} - {:^35}  Processing: {:4d} / {:4d}".format(state, video_name, len(image_feature), len(video_image_list)), end='\r')
		image_feature = np.concatenate(image_feature[:], axis=0)
	
		np.save(os.path.join(save_feature_path, video_name), image_feature)
	print('\n')

def get_batch_num(video_list, action_size, batch_size, state):
	path = os.path.join(video_path, state)
	batch_num = []
	for video_name in video_list:
		image_num = len(getVideoList(os.path.join(path, video_name)))
		batch_num.append(math.floor((image_num - action_size + 1) / batch_size))

	return np.array(batch_num)

def load_label(video_name, state):
	path = os.path.join(label_path, state)
	file_path = os.path.join(path, "{}.txt".format(video_name))
	labels = np.loadtxt(file_path).astype(np.int32)
	
	return labels

def process_sliding_window(input_array, window_size, sliding_size):
	output_array = []
	for index in range(0, len(input_array) - window_size + 1, sliding_size):
		output_array.append(input_array[index:index+window_size])

	return np.array(output_array)

def load_data_generator(video_list, action_size, batch_size, batch_num, state):
	file_order = np.arange(0, len(video_list))

	path = os.path.join(video_feature_path, state)

	start_index = 0
	while True:
		# Shuffle data
		if state == 'train':
			np.random.shuffle(file_order)

		while start_index < len(video_list):
			features = np.load(os.path.join(path, "{}.npy".format(video_list[file_order[start_index]])))
			labels = load_label(video_list[file_order[start_index]], state)
			labels = labels.reshape(labels.shape + (1,))
			# Process sliding window
			features = process_sliding_window(features, action_size, 1)
			labels = process_sliding_window(labels, action_size, 1)
			# image shuffle
			image_order = np.arange(0, len(features))
			if state == 'train':
				np.random.shuffle(image_order)

			for batch_index in range(batch_num[file_order[start_index]]):
				X = features[image_order[batch_index*batch_size:(batch_index+1)*batch_size]]
				Y = labels[image_order[batch_index*batch_size:(batch_index+1)*batch_size]]
				yield(X, Y)
			start_index += 1
		start_index = 0

if __name__ == '__main__':
	train_video_list = getVideoList(os.path.join(video_path, 'train'))
	valid_video_list = getVideoList(os.path.join(video_path, 'valid'))

	# extract_features_model = Inception_extract_features_model()
	# extract_features_model.summary()

	# Pre-processing: extract features from video frames by pre-trained model
	# Extract_features(extract_features_model, valid_video_list, 'valid')
	# Extract_features(extract_features_model, train_video_list, 'train')

	# Load LSTM weights from HW5_problem2
	lstm = LSTM_model()
	lstm.summary()
	lstm.load_weights(lstm_weights_path, by_name=True)

	epochs = 100
	action_image_num = 10
	train_batch_size = 4
	valid_batch_size = 1
	valid_batch_num = get_batch_num(valid_video_list, action_image_num, valid_batch_size, 'valid')
	train_batch_num = get_batch_num(train_video_list, action_image_num, train_batch_size, 'train')

	valid_generator = load_data_generator(valid_video_list, action_image_num, valid_batch_size, valid_batch_num, 'valid')
	train_generator = load_data_generator(train_video_list, action_image_num, train_batch_size, train_batch_num, 'train')

	lstm.compile(optimizer=Adam(lr=1e-7), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	checkpoint = ModelCheckpoint('p3_lstm_model.h5', monitor='val_loss', save_best_only=True)

	history = lstm.fit_generator(train_generator,
					steps_per_epoch=np.sum(train_batch_num), 
					epochs=epochs,   
					validation_data=valid_generator,
					validation_steps=np.sum(valid_batch_num),
					callbacks=[checkpoint])

	# Record loss and accuracy
	curve_history = dict()
	curve_history['loss'] = history.history['loss']
	curve_history['accuracy'] = history.history['acc']
	curve_history['val_loss'] = history.history['val_loss']
	curve_history['val_accuracy'] = history.history['val_acc']
	
	with open('p3_curve_history.pickle', 'wb') as outfile:
		pickle.dump(curve_history, outfile)
