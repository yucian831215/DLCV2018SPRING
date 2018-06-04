import skvideo.io
import skimage.transform
import csv
import collections
import os
import pickle
import math
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3 as Inception

train_ground_truth_path = './HW5_data/TrimmedVideos/label/gt_train.csv'
valid_ground_truth_path = './HW5_data/TrimmedVideos/label/gt_valid.csv'
video_path = './HW5_data/TrimmedVideos/video'
model_path = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
video_feature_path = './features'

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.accuracy = []

	def on_batch_end(self, batch, logs={}):
		self.loss.append(logs.get('loss', 0))
		self.accuracy.append(logs.get('acc', 0))

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
            frames.append(frame)
        else:
            continue

    return np.array(frames).astype(np.uint8)

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

def normalize_frame(frame):
	output_frame = frame / 255
	return output_frame

def save_video_features(video_index, features, state):
	name = "Video_{}".format(str(video_index).zfill(4))
	path = os.path.join(video_feature_path, state)
	np.save(os.path.join(path, name), features)
 
def Inception_extract_features_model():
	# Build conv and pooling layers according to inception
	model = Inception(include_top=False, weights=None, input_shape=(240, 320, 3))
	model.load_weights(model_path)
	Inception_inputs = Input(shape=(240, 320, 3))
	Inception_outputs = model(Inception_inputs)
	model = Model(Inception_inputs, GlobalAveragePooling2D()(Inception_outputs))

	return model

def Fully_connected_layers(state):
	if state == 'average':
		features_input = Input(shape=(2048, ))
	elif state == 'concatenate':
		features_input = Input(shape=(4 * 2048, ))

	x = Dense(512, activation='relu', name='fc1')(features_input)
	x = Dropout(0.5)(x)

	x = Dense(512, activation='relu', name='fc2')(x)
	x = Dropout(0.5)(x)

	x = Dense(11, activation='softmax', name='predictions')(x)

	model = Model(features_input, x)

	return model

def Extract_features(model, video_list, state):
	path = os.path.join(video_path, state)
	for index in range(len(video_list['Video_index'])):
		video_frame = readShortVideo(path, video_list['Video_category'][index], video_list['Video_name'][index])
		video_frame = normalize_frame(video_frame)
		if len(video_frame) > 10:
			batch_num = int(math.ceil(len(video_frame) / 10))
			video_feature = []
			for batch_index in range(batch_num):
				if batch_index + 1 < batch_num:
					video_feature.append(model.predict(video_frame[batch_index * 10 : (batch_index + 1) * 10]))
				else:
					video_feature.append(model.predict(video_frame[batch_index * 10 : ]))
			video_feature = np.concatenate(video_feature[:], axis=0)
		else:
			video_feature = model.predict(video_frame)

		save_video_features(video_list['Video_index'][index], video_feature, state)
		print("{} - Index: {}, Feature_Size: {}, Frame_Size: {}.".format(state, video_list['Video_index'][index], video_feature.shape, video_frame.shape), end='\r')
	print('\n')

def Feature_strategy(features, state):
	if state == 'average':
		output_feature = np.mean(features, axis=0)
		return output_feature
	elif state == 'concatenate':
		zero_padding = np.zeros(2048)
		output_feature = []
		if len(features) >= 4:
			# Start feature
			output_feature.append(features[0])
			# Middle feature
			middle_point = int(math.ceil((len(features) - 1) / 2))
			output_feature.append(features[middle_point - 1])
			output_feature.append(features[middle_point])
			# End feature
			output_feature.append(features[len(features) - 1])
		elif len(features) == 3:
			# Start feature
			output_feature.append(features[0])
			# Middle feature
			middle_point = int(math.ceil((len(features) - 1) / 2))
			output_feature.append(zero_padding)
			output_feature.append(features[middle_point])
			# End feature
			output_feature.append(features[len(features) - 1])
		elif len(features) == 2:
			# Start feature
			output_feature.append(features[0])
			# Middle feature
			output_feature.append(zero_padding)
			output_feature.append(zero_padding)
			# End feature
			output_feature.append(features[len(features) - 1])
		elif len(features) == 1:
			# Start feature
			output_feature.append(features[0])
			output_feature.append(zero_padding)
			output_feature.append(zero_padding)
			output_feature.append(zero_padding)
		
		output_feature = np.concatenate(output_feature[:], axis=0)
		return output_feature


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
			path = os.path.join(video_feature_path, state)
			features = np.load(os.path.join(path, "Video_{}.npy".format(str(data_order[start_index + index]).zfill(4))))
			X.append(Feature_strategy(features, "average"))
			Y.append(int(video_list['Action_labels'][data_order[start_index + index] - 1]))

		X = np.array(X)
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
	
	# extract_features_model = Inception_extract_features_model()
	# extract_features_model.summary()

	# Pre-processing: extract features from video frames by pre-trained model
	# Extract_features(extract_features_model, valid_video_list, 'valid')
	# Extract_features(extract_features_model, train_video_list, 'train')
	
	#---------------------------------------------------#
	# Train fully connected layers (Use fit_generator)  #
	#---------------------------------------------------#
	train_data_size = len(train_video_list['Video_index'])
	valid_data_size = len(valid_video_list['Video_index'])
	train_batch_size = 4
	valid_batch_size = 1
	train_batch_num = int(train_data_size / train_batch_size)
	valid_batch_num = int(valid_data_size / valid_batch_size)

	epochs = 100

	valid_generator = load_data_batch(valid_video_list, valid_batch_size, valid_data_size, "valid")
	train_generator = load_data_batch(train_video_list, train_batch_size, train_data_size, "train")
	
	fc_model = Fully_connected_layers('average')
	fc_model.summary()

	fc_model.compile(optimizer=Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	checkpoint = ModelCheckpoint('p1_fc_model.h5', monitor='val_loss', save_best_only=True)

	history = LossHistory()

	val_history = fc_model.fit_generator(train_generator,
					steps_per_epoch=train_batch_num, 
					epochs=epochs,   
					validation_data=valid_generator,
					validation_steps=valid_batch_num,
					callbacks=[checkpoint, history])
	
	# Record loss and accuracy
	curve_history = dict()
	curve_history['loss'] = history.loss
	curve_history['accuracy'] = history.accuracy
	curve_history['val_loss'] = val_history.history['val_loss']
	curve_history['val_accuracy'] = val_history.history['val_acc']
	
	with open('p1_curve_history.pickle', 'wb') as outfile:
		pickle.dump(curve_history, outfile)
