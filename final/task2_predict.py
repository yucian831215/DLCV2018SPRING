import os
import json
import math
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from skimage import io
from keras.layers import *
from keras.models import Model, load_model

input_path = ''
output_path = ''
pathModel = 'mini_inception.h5'
pathTable = './Facebank_info/matchTable.jsdb'
matchTable = dict()

def load_matchTable():
	table = dict()
	with open(pathTable, 'r') as outfile:
		table = json.load(outfile)

	return table

def normalize_image(img):
		output_img = img / 127.5 - 1
		return output_img

def listFileName(path):
	file_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
	file_list.sort()

	return file_list

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputs', help='input dataset directory', type=str)
	parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output',type=str)
	args = parser.parse_args()

	input_path = args.inputs
	output_path = args.outputs

	matchTable = load_matchTable()
	new_matchTable = {v : k for k, v in matchTable.items()}

	input_list = listFileName(input_path)
	output_list = []

	print(" ----- Load {} model ----".format(pathModel))
	model = load_model(pathModel)
	model.summary()

	print(" ----- In predict program -----")
	for i in range(len(input_list)):
		print("Processing face recognition - {:>4d} / {}".format(i + 1, len(input_list)), end='\r')
		path = os.path.join(input_path, input_list[i])
		img = normalize_image(io.imread(path))
		input_img = img.reshape((1, ) + img.shape)
		pred = model.predict(input_img)
		output_label = np.argmax(pred[0])
		output_list.append(new_matchTable[output_label])

	print("\n ---- Predict program Done ! ---- ")

	with open(os.path.join(output_path, 'result.csv'), 'w') as outfile:
		outfile.write('id,ans\n')
		for i in range(len(input_list)):
			imgId = str(int(input_list[i].split('.')[0]))
			imgAns = str(output_list[i])
			outfile.write('{},{}\n'.format(imgId, imgAns))
	outfile.close()

	print("Save output in {}".format(os.path.join(output_path, 'result.csv')))
	




