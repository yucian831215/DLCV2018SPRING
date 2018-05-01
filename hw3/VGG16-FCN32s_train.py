import os
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
# import tensorflow as tf

# from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, Activation, Cropping2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam 
# from keras.layers.normalization import BatchNormalization
from skimage import io

weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# weights_path = './model_5_picture.h5'
train_path = './hw3-train-validation/train'
# train_path = './Try2'
validation_path = './hw3-train-validation/validation'
# validation_path = './Try2'

def read_image_name(path):
	# Divide image into two parts: mask, sat
	sat_file_list = [file for file in os.listdir(path) if file.endswith('sat.jpg')]
	sat_file_list.sort()

	mask_file_list = [file for file in os.listdir(path) if file.endswith('mask.png')]
	mask_file_list.sort()

	return sat_file_list, mask_file_list

def process_image(path, input_name, output_name):
	# Show numpy array of sat_image
	# print(input_name, output_name)
	sat_image = io.imread(os.path.join(path, input_name))
	# Sat_image normalize
	# sat_image = sat_image / 255.0
	sat_image = sat_image / 127.5 - 1

	mask_image = np.zeros((512, 512, 7))
	
	# Show the label of mask_image (512, 512, 7)
	mask_img = io.imread(os.path.join(path, output_name))
	mask_img = (mask_img >= 128).astype(int)
	mask_img = 4 * mask_img[:, :, 0] + 2 * mask_img[:, :, 1] + mask_img[:, :, 2]

	for row in range(mask_img.shape[0]):
		for col in range(mask_img.shape[1]):
			if mask_img[row][col] == 3:
				mask_image[row][col][0] = 1
			elif mask_img[row][col] == 6:
				mask_image[row][col][1] = 1
			elif mask_img[row][col] == 5:
				mask_image[row][col][2] = 1
			elif mask_img[row][col] == 2:
				mask_image[row][col][3] = 1
			elif mask_img[row][col] == 1:
				mask_image[row][col][4] = 1
			elif mask_img[row][col] == 7:
				mask_image[row][col][5] = 1
			else:
				mask_image[row][col][6] = 1

	return sat_image, mask_image

def load_train_batch_data(input_name, output_name, batch_size, data_size, path=train_path):
	# Shuffle dataset
	data_order = np.arange(data_size)
	np.random.shuffle(data_order)
	# print(data_order)

	start_index = 0
	while True:
		input_X = []
		output_Y = []
		for img_index in range(batch_size):
			# Load images: sat, mask
			img_X, img_Y = process_image(path, input_name[data_order[start_index + img_index]], output_name[data_order[start_index + img_index]])
			# print(path, input_name[data_order[start_index + img_index]], output_name[data_order[start_index + img_index]])
			input_X.append(img_X)
			output_Y.append(img_Y)

		input_X = np.array(input_X)
		output_Y = np.array(output_Y)
		yield(input_X, output_Y)

		start_index += batch_size
		if start_index >= data_size:
			start_index = 0
			np.random.shuffle(data_order)
			# print(data_order)

def load_validation_batch_data(input_name, output_name, batch_size, data_size, path=validation_path):
	start_index = 0
	while True:
		input_X = []
		output_Y = []
		for img_index in range(batch_size):
			# Load images: sat, mask
			img_X, img_Y = process_image(path, input_name[start_index + img_index], output_name[start_index + img_index])
			# print(path, input_name[start_index + img_index], output_name[start_index + img_index])
			input_X.append(img_X)
			output_Y.append(img_Y)

		input_X = np.array(input_X)
		output_Y = np.array(output_Y)
		yield(input_X, output_Y)

		start_index += batch_size
		if start_index >= data_size:
			start_index = 0

# def mean_iou(y_true, y_pred):
# 	score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 7)
# 	K.get_session().run(tf.local_variables_initializer())
# 	with tf.control_dependencies([up_opt]):
# 		score = tf.identity(score)
# 	# return tf.metrics.mean_iou(y_true, y_pred, 7)[0]
# 	return score

def VGG16_FCN32s_model(input_height, input_weight, n_classes):
	# Build conv and pooling layers according to VGG16
	img_input = Input(shape=(input_height, input_weight, 3))
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	# x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	# x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	# x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	# x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	# x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	x = Conv2D(512, (7, 7), activation='relu', padding='same', name='block6_conv1')(x)
	x = Dropout(0.5)(x)

	x = Conv2D(512, (1, 1), activation='relu', padding='same', name='block7_conv1')(x)
	x = Dropout(0.5)(x)

	x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', name='block8_conv1')(x)
	# x = Conv2D(n_classes, (1, 1), activation='relu', padding='same', name='block8_conv1')(x)
	# x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='block8_conv1')(x)

	x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False, name='block9_transpose')(x)
	# x = UpSamplin0g2D((32, 32))(x)

	x = Cropping2D(cropping=((16,16),(16,16)))(x)

	x = Activation('softmax')(x)
	
	model = Model(img_input, x)
	
	# model_shape = model.output_shape
	# print(model_shape)

	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--train', help='train dataset directory', type=str)
	parser.add_argument('-v', '--validation', help='validation dataset directory', type=str)
	args = parser.parse_args()

	train_path = args.train
	validation_path = args.validation
	
	model = VGG16_FCN32s_model(512, 512, 7)
	# Load pretrained weights from VGG16
	model.load_weights(weights_path, by_name=True)

	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['categorical_accuracy'])

	model.summary()

	# # for layer in model.layers[:19]:
	# # 	layer.trainable = False
		
	checkpoint = ModelCheckpoint('fcn32s_model.h5', monitor='val_loss', save_best_only=True)

	train_input_name, train_output_name = read_image_name(train_path)
	validation_input_name, validation_output_name = read_image_name(validation_path)

	train_data_number = len(train_input_name)
	validation_data_number = len(validation_input_name)
	# print(train_data_number)
	batch_size = 1
	train_batch_number = int(train_data_number / batch_size)
	validation_batch_number = int(validation_data_number / 1)

	epochs = 30

	train_generator = load_train_batch_data(train_input_name, train_output_name, batch_size, train_data_number)
	validation_generator = load_validation_batch_data(validation_input_name, validation_output_name, 1, validation_data_number)

	# for i in range(10*train_batch_number):
	# 	next(train_generator)

	model.fit_generator(train_generator,
				steps_per_epoch=train_batch_number, 
				epochs=epochs,   
				validation_data=validation_generator,
				validation_steps=validation_batch_number,
				callbacks=[checkpoint])	

	# for epoch in range(epochs):
	# 	print("Eopch: ", epoch)
	# 	for img_index in range(len(input_path)):
	# 		# start_time = time.time()
	# 		input_X, output_Y = process_image(input_path, output_path, img_index)
	# 		# print(input_X[img_index])
	# 		# print("File: ", input_path[img_index], "Time: ", time.time() - start_time, "Output: ", output_Y[0][0])
	# 		model.fit(input_X.reshape((1, ) + input_X.shape), output_Y.reshape((1, ) + output_Y.shape), epochs=1, batch_size=4, callbacks=[checkpoint])

	# # print(input_X.shape, output_Y.shape)
	# # print(output_Y[0][0])


