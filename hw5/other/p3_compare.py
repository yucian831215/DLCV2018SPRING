import numpy as np
import os
import argparse

ground_truth_path = ''
predict_path = ''

def load_list(path):
	result_list = [result for result in os.listdir(path)]
	result_list.sort()

	return result_list

def getAccuracy(predict_file, ground_truth_file):
	pfile_path = os.path.join(predict_path, predict_file)
	gfile_path = os.path.join(ground_truth_path, ground_truth_file)

	predict_label = np.loadtxt(pfile_path).astype(np.int32)
	ground_truth_label = np.loadtxt(gfile_path).astype(np.int32)

	num_data = len(predict_label)
	num_correct_label = 0
	for index in range(len(ground_truth_label)):
		if predict_label[index] == ground_truth_label[index]:
			num_correct_label += 1
	accuracy = num_correct_label / num_data

	return accuracy

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--predict', help='predict txt path', type=str)
	parser.add_argument('-g', '--ground', help='ground_truth txt path', type=str)
	args = parser.parse_args()

	predict_path = args.predict
	ground_truth_path = args.ground

	predict_list = load_list(predict_path)
	ground_truth_list = load_list(ground_truth_path)

	for index in range(len(predict_list)):
		accuracy = getAccuracy(predict_list[index], ground_truth_list[index])
		print("{name} - Accuracy: {a:.3f} %".format(name=predict_list[index].split('.')[0], a=accuracy * 100))

