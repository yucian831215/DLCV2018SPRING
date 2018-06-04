import skvideo.io
import skimage.transform
import csv
import collections
import os
import pickle
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras.applications.inception_v3 import InceptionV3 as Inception

valid_ground_truth_path = './HW5_data/TrimmedVideos/label/gt_valid.csv'
video_path = './HW5_data/TrimmedVideos/video/valid'
inception_model_path = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
fc_model_path = './model/p1_fc_model.h5'
ouput_path = './output'

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

def Inception_extract_features_model():
	# Build conv and pooling layers according to inception
	model = Inception(include_top=False, weights=None, input_shape=(240, 320, 3))
	model.load_weights(inception_model_path)
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

	return video_feature

def Feature_strategy(features, state):
	if state == 'average':
		output_feature = np.mean(features, axis=0)
		return output_feature

def Action_Recognition(inception_model, fc_model, video_frame, batch_size):
	# Get features by inception_model @ each input dim of frame is 10
	if len(video_frame) > batch_size:
		batch_num = int(math.ceil(len(video_frame) / batch_size))
		video_feature = []
		for batch_index in range(batch_num):
			if batch_index + 1 < batch_num:
				video_feature.append(inception_model.predict(video_frame[batch_index * batch_size : (batch_index + 1) * batch_size]))
			else:
				video_feature.append(inception_model.predict(video_frame[batch_index * batch_size : ]))
		video_feature = np.concatenate(video_feature[:], axis=0)
	else:
		video_feature = inception_model.predict(video_frame)

	# Use the feature strategy
	rep_feature = Feature_strategy(video_feature, 'average')
	rep_feature = rep_feature.reshape((1, ) + rep_feature.shape)

	# Use fc_model to get the action label
	label_softmax = fc_model.predict(rep_feature)
	label = np.argmax(label_softmax)

	return label

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputs', help='input dataset directory', type=str)
	parser.add_argument('-c', '--input_csv', help='the informatino about input dataset (csv)', type=str)
	parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output',type=str)
	args = parser.parse_args()

	video_path = args.inputs
	valid_ground_truth_path = args.input_csv
	output_path = args.outputs

	valid_video_list = getVideoList(valid_ground_truth_path)

	extract_features_model = Inception_extract_features_model()
	extract_features_model.summary()

	fc_model = Fully_connected_layers('average')
	fc_model.load_weights(fc_model_path)
	fc_model.summary()

	labels = []
	for index in range(len(valid_video_list['Video_index'])):
		print('{index_now:3d} / {num_video:3d} - Processing {video_name:<45}'.format(index_now=index+1,
																			   num_video=len(valid_video_list['Video_index']),
			                                                                   video_name=valid_video_list['Video_name'][index]), end='\r')
		# Get the frames from video
		video_frame = readShortVideo(video_path, valid_video_list['Video_category'][index], valid_video_list['Video_name'][index])
		video_frame = normalize_frame(video_frame)
		labels.append(Action_Recognition(extract_features_model, fc_model, video_frame, 10))
	print('\n')

	labels = np.array(labels)
	labels = labels.reshape(labels.shape + (1,))
	np.savetxt(os.path.join(output_path, 'p1_valid.txt'), labels, fmt='%d')
		
