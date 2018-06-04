import os
import pickle
import math
import keras
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from skimage import io

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras.applications.inception_v3 import InceptionV3 as Inception

video_path = './HW5_data/FullLengthVideos/videos/valid'
inception_model_path = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
lstm_model_path = './model/p3_lstm_model.h5'
output_path = './output'

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

def Extract_features(inception_model, video_name):
	video_image_path = os.path.join(video_path, video_name)
	video_image_list = getVideoList(video_image_path)
	image_feature = []
	
	index_now = 0
	# Extract image feature by Inception model
	for image in video_image_list:
		print('{index:4d} / {num_image:4d} Extract features processing: {video_name}'.format(index=index_now+1,
																				num_image=len(video_image_list),
																				video_name=video_name), end='\r')
		image = io.imread(os.path.join(video_image_path, image))
		image = normalize_image(image).reshape((1, ) + image.shape)
		image_feature.append(inception_model.predict(image))
		index_now += 1
	image_feature = np.concatenate(image_feature[:], axis=0)
	print('\n')
	return image_feature

def process_sliding_window(input_array, window_size, sliding_size):
	output_array = []
	for index in range(0, len(input_array) - window_size + 1, sliding_size):
		output_array.append(input_array[index:index+window_size])

	return np.array(output_array)

def Action_Detection(lstm_model, input_feature, num_image):
	print("Processing LSTM ...")
	label_vote = np.zeros((num_image, 11)).astype(np.int32)
	for index in range(len(input_feature)):
		input_X = input_feature[index].reshape((1, ) + input_feature[index].shape)
		labels_softmax = lstm_model.predict(input_X)[0]
		labels = np.argmax(labels_softmax, axis=-1)
		for label_index in range(len(labels)):
			label_vote[index + label_index][labels[label_index]] += 1
	result_label = np.argmax(label_vote, axis=-1)

	return result_label

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputs', help='input dataset directory', type=str)
	parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output',type=str)
	args = parser.parse_args()

	video_path = args.inputs
	output_path = args.outputs

	valid_video_list = getVideoList(video_path)

	extract_features_model = Inception_extract_features_model()
	extract_features_model.summary()

	lstm_model = LSTM_model()
	lstm_model.load_weights(lstm_model_path)
	lstm_model.summary()

	action_size = 10

	# for video_name in valid_video_list:
	for video_name in valid_video_list:
		input_feature = Extract_features(extract_features_model,video_name)
		sliding_feature = process_sliding_window(input_feature, action_size, 1)
		labels = Action_Detection(lstm_model, sliding_feature, len(input_feature))
		labels = labels.reshape(labels.shape + (1,))
		np.savetxt(os.path.join(output_path, "{video_category}.txt".format(video_category=video_name)), labels, fmt='%d')
