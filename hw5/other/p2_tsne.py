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
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from sklearn.manifold import TSNE

from keras.applications.inception_v3 import InceptionV3 as Inception

valid_ground_truth_path = './HW5_data/TrimmedVideos/label/gt_train.csv'
video_path = './HW5_data/TrimmedVideos/video/train'
inception_model_path = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
lstm_model_path = './model/p2_lstm_model.h5'
output_path = './output'

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

def LSTM_model():
	# Design LSTM model
	LSTM_inputs = Input(shape=(None, 2048))

	x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, name='lstm_1'), merge_mode='concat')(LSTM_inputs)
	x = LSTM(256, dropout=0.2, name='lstm_2')(x)

	model = Model(LSTM_inputs, x)

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

def plot_tSNE(input_vector, labels, state):
	input_vectors = input_vector.astype(np.float64)
	embedded_vectors = TSNE(n_components=2, random_state=0).fit_transform(input_vectors)
	point_X = embedded_vectors[:,0]
	point_Y = embedded_vectors[:,1]

	plt.title("TSNE of {} features".format(state))
	plt.scatter(x=point_X, y=point_Y, c=labels, s=[5])
	plt.axis('off')

	plt.savefig(os.path.join(output_path, 'tSNE_{}.jpg'.format(state)))
	plt.close()

if __name__ == '__main__':
	valid_video_list = getVideoList(valid_ground_truth_path)

	extract_features_model = Inception_extract_features_model()
	extract_features_model.summary()

	lstm_model = LSTM_model()
	lstm_model.load_weights(lstm_model_path, by_name=True)
	lstm_model.summary()

	cnn_feature = []
	lstm_feature = []
	for index in range(len(valid_video_list['Video_index'])):
		print('{index_now:3d} / {num_video:3d} - Processing {video_name:<45}'.format(index_now=index+1,
																			   num_video=len(valid_video_list['Video_index']),
			                                                                   video_name=valid_video_list['Video_name'][index]), end='\r')
		# Get the frame from video
		video_frame = readShortVideo(video_path, valid_video_list['Video_category'][index], valid_video_list['Video_name'][index])
		video_frame = normalize_frame(video_frame)

		# Get the feature from inception
		feature = extract_features_model.predict(video_frame)

		# Get the feature by Feature strategy
		cnn_feature.append(Feature_strategy(feature, 'average'))

		# Get the feature by LSTM
		feature = feature.reshape((1, ) + feature.shape)
		lstm_feature.append(lstm_model.predict(feature)[0])
	print('\n')

	cnn_feature = np.array(cnn_feature)
	lstm_feature = np.array(lstm_feature)
	video_labels = np.array(valid_video_list['Action_labels'])

	# tSNE of cnn_feature
	print("Procssing tSNE of CNN feature...")
	plot_tSNE(cnn_feature, video_labels, 'CNN')
	# tSNE of lstm_feature
	print("Procssing tSNE of LSTM feature...")
	plot_tSNE(lstm_feature, video_labels, 'LSTM')


	