import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from skimage import io

input_data_path = './hw3-train-validation/validation/'
# input_data_path = './Try1/'
output_data_path = './output/'
# output_data_path = './Try1/'
model_path = './fcn32s_model.h5'

def read_image_name(path):
	# Divide image into two parts: mask, sat
	sat_file_list = [file for file in os.listdir(path) if file.endswith('sat.jpg')]
	sat_file_list.sort()

	return sat_file_list

def process_image(img):
	color_result = np.argmax(img[0], axis=-1)

	output_result = np.zeros((512, 512, 3), dtype=np.uint8)
	
	# for row in range(color_result.shape[1]):
	# 	for col in range(color_result.shape[2]):
	# 		if color_result[0][row][col] == 0:
	# 			output_result[row][col][0] = 0
	# 			output_result[row][col][1] = 255
	# 			output_result[row][col][2] = 255
	# 		elif color_result[0][row][col] == 1:
	# 			output_result[row][col][0] = 255
	# 			output_result[row][col][1] = 255
	# 			output_result[row][col][2] = 0
	# 		elif color_result[0][row][col] == 2:
	# 			output_result[row][col][0] = 255
	# 			output_result[row][col][1] = 0
	# 			output_result[row][col][2] = 255
	# 		elif color_result[0][row][col] == 3:
	# 			output_result[row][col][0] = 0
	# 			output_result[row][col][1] = 255
	# 			output_result[row][col][2] = 0
	# 		elif color_result[0][row][col] == 4:
	# 			output_result[row][col][0] = 0
	# 			output_result[row][col][1] = 0
	# 			output_result[row][col][2] = 255
	# 		elif color_result[0][row][col] == 5:
	# 			output_result[row][col][0] = 255
	# 			output_result[row][col][1] = 255
	# 			output_result[row][col][2] = 255
	# 		elif color_result[0][row][col] == 6:
	# 			output_result[row][col][0] = 0
	# 			output_result[row][col][1] = 0
	# 			output_result[row][col][2] = 0

	output_result[color_result == 0] = [0, 255, 255]
	output_result[color_result == 1] = [255, 255, 0]
	output_result[color_result == 2] = [255, 0, 255]
	output_result[color_result == 3] = [0, 255, 0]
	output_result[color_result == 4] = [0, 0, 255]
	output_result[color_result == 5] = [255, 255, 255]
	output_result[color_result == 6] = [0, 0, 0]
				
	return output_result

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', help='model path', type=str)
	parser.add_argument('-i', '--inputs', help='input dataset directory', type=str)
	parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output', type=str)
	args = parser.parse_args()

	model_path = args.model
	input_data_path = args.inputs
	output_data_path = args.outputs

	print("Load model...")
	model = load_model(model_path)

	input_name = read_image_name(input_data_path)

	if not os.path.isdir(output_data_path):
		os.mkdir(output_data_path)

	for i in input_name:
		output_name = i.replace("sat.jpg", "mask.png")
		# print("\r", i, output_name)
		print("\rProcessing: {}  →  {}".format(i, output_name), end="")
		img = io.imread(os.path.join(input_data_path, i))
		img = img.reshape((1, ) + img.shape) / 127.5 - 1
		result_img = model.predict(img)
		output_result = process_image(result_img)
		plt.imsave(os.path.join(output_data_path, output_name), output_result)

	print("\nDone!")
	# img = io.imread("/home/user/Desktop/DLCV/HW3/hw3-train-validation/train/0003_sat.jpg")
	# img = img.reshape((1, ) + img.shape)/255

	# result_img = model.predict(img)
	

	# output_result = cv2.cvtColor(output_result, cv2.COLOR_BGR2RGB)
	# cv2.imshow("123", output_result)
	# cv2.waitKey()
	# plt.imshow(output_result)
	# plt.show()
