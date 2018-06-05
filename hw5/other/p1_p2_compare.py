import numpy as np
import argparse
import csv
import collections

ground_truth_path = ''
predict_path = ''

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--predict', help='predict txt path', type=str)
	parser.add_argument('-g', '--ground', help='ground_truth txt path', type=str)
	args = parser.parse_args()

	predict_path = args.predict
	ground_truth_path = args.ground

	predict_label = np.loadtxt(predict_path).astype(np.int32)

	num_data = len(predict_label)
	ground_truth_info = getVideoList(ground_truth_path)
	ground_truth_label = np.array(ground_truth_info['Action_labels']).astype(np.int32)

	num_correct_label = 0
	for index in range(num_data):
		if predict_label[index] == ground_truth_label[index]:
			num_correct_label += 1

	accuracy = num_correct_label / num_data

	print("Accuracy: {:.3f} %".format(accuracy * 100))

	
