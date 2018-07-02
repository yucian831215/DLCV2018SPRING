import json
import os
import re

import numpy as np

pathDataset = './dlcv_final_2_dataset'
pathInfo = './Facebank_info'

def LabelTable():
	matchTable = dict()
	label = 0
	path = os.path.join(pathDataset, 'train_id.txt')
	with open(path, 'r') as outfile:
		for outline in outfile:
			content = outline.split()
			if not content[1] in matchTable.keys():
				matchTable[content[1]] = label
				label += 1
	pathTable = os.path.join(pathInfo, 'matchTable.jsdb')
	with open(pathTable, 'w') as outfile:
		json.dump(matchTable, outfile)

def Relabel_data(state):
	matchTable = dict()
	pathTable = os.path.join(pathInfo, 'matchTable.jsdb')
	with open(pathTable, 'r') as outfile:
		matchTable = json.load(outfile)

	labelData = dict()
	pathId = os.path.join(pathDataset, "{}_id.txt".format(state))
	with open(pathId, 'r') as outfile:
		for outline in outfile:
			content = outline.split()
			label = matchTable[content[1]]
			labelData[content[0]] = label

	pathRelabel = os.path.join(pathInfo, "{}_relabel.jsdb".format(state))
	with open(pathRelabel, 'w') as outfile:
		json.dump(labelData, outfile)

if __name__ == '__main__':
	LabelTable()

	Relabel_data("train")
	Relabel_data("val")