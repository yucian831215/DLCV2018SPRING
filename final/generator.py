import math

import numpy as np

from skimage import io
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

class dataGenerator(Sequence):
	def __init__(self, datalist, labellist, batchSize=16, shuffle=False, aug=False):
		self.datalist = datalist
		self.labellist = labellist
		self.batchSize = batchSize
		self.shuffle = shuffle
		self.indexlist = np.arange(len(self.datalist))
		self.aug = aug
		self.imageAugment = ImageDataGenerator(rotation_range = 30,
						horizontal_flip=True, width_shift_range = 0.25,
						height_shift_range = 0.25,)

	def __len__(self):
		return int(math.ceil(len(self.datalist) / self.batchSize))

	def __getitem__(self, index):
		startIndex = index * self.batchSize
		endIndex = min((index + 1) * self.batchSize, len(self.datalist))

		indexlist = self.indexlist[startIndex : endIndex]
		X, Y = self.__dataGenerator(indexlist)

		return X, Y

	def on_epoch_end(self):
		if self.shuffle:
			np.random.shuffle(self.indexlist)

	def normalize_image(self, img):
		output_img = img / 127.5 - 1
		return output_img

	def __dataGenerator(self, indexlist):
		Label = [self.labellist[index] for index in indexlist]
		ImgPath = [self.datalist[index] for index in indexlist]

		DataInput = [self.normalize_image(io.imread(path)) for path in ImgPath]

		if self.aug:
			DataInput, Label = next(self.imageAugment.flow(np.array(DataInput), np.array(Label),
							batch_size=self.batchSize, shuffle=False))

		return np.array(DataInput), np.array(Label)
