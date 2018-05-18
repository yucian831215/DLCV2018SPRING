import os
import pickle
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from skimage import io
from sklearn.manifold import TSNE
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.metrics import mean_squared_error

test_path = './hw4_data/test/'
weight_path = './vae_model.h5'
loss_path = './VAE_loss_history.pickle'
label_path = './hw4_data/test.csv'
output_path = './'

intermediate_dim = 128
latent_dim = 512

def loss_graph():
	with open(loss_path, 'rb') as outfile:
		loss_history = pickle.load(outfile)
	plt.figure(figsize=(16,6))

	plt.subplot(1, 2, 1)
	loss_history_MSE_X = np.arange(len(loss_history['MSE']))
	loss_history_MSE_Y = np.array(loss_history['MSE'])
	smooth = InterpolatedUnivariateSpline(loss_history_MSE_X, loss_history_MSE_Y)
	loss_history_MSE_X_smooth = np.linspace(loss_history_MSE_X.min(), loss_history_MSE_X.max(), num=1000).astype(np.int32)
	loss_history_MSE_Y_smooth = smooth(loss_history_MSE_X_smooth)
	plt.plot(loss_history_MSE_X, loss_history_MSE_Y, color='mistyrose', linewidth=1)
	plt.plot(loss_history_MSE_X_smooth, loss_history_MSE_Y_smooth, color='r', linewidth=1.5)
	plt.xlabel("Training Steps")
	plt.title("MSE")

	plt.subplot(1, 2, 2)
	loss_history_KL_X = np.arange(len(loss_history['KL']))
	loss_history_KL_Y = np.array(loss_history['KL'])
	smooth = InterpolatedUnivariateSpline(loss_history_KL_X, loss_history_KL_Y)
	loss_history_KL_X_smooth = np.linspace(loss_history_KL_X.min(), loss_history_KL_X.max(), num=1000).astype(np.int32)
	loss_history_KL_Y_smooth = smooth(loss_history_KL_X_smooth)
	plt.plot(loss_history_KL_X, loss_history_KL_Y, color='mistyrose', linewidth=1)
	plt.plot(loss_history_KL_X_smooth, loss_history_KL_Y_smooth, color='r', linewidth=1.5)
	plt.xlabel("Training Steps")
	plt.title("KLD")

	plt.savefig(os.path.join(output_path, "fig1_2.jpg"))
	plt.close()

def reconstructed_imgage(image_list, path=test_path):
	vae_model = VAE_model(64, 64, 3)
	vae_model.load_weights(weight_path, by_name=True)

	test_image_num = 10
	col_num = test_image_num + 1

	plt.figure(figsize=(16,4))
	plt.subplot(2, col_num, 1)
	plt.axis('off')
	plt.text(0, 0.5, "Input", size=16)

	plt.subplot(2, col_num, col_num + 1)
	plt.axis('off')
	plt.text(-0.5, 0.5, "Reconstruct", size=16)

	for img_index in range(test_image_num):
		# Original image
		img = io.imread(os.path.join(path, image_list[img_index]))
		plt.subplot(2, col_num, img_index + 2)
		plt.axis('off')
		plt.imshow(img)
		# Predicted image
		input_img = normalize_image(img.reshape((1, ) + img.shape))
		result_img = vae_model.predict(input_img)
		output_result = process_image(result_img[0])
		plt.subplot(2, col_num, col_num + img_index + 2)
		plt.axis('off')
		plt.imshow(output_result)

	plt.savefig(os.path.join(output_path, "fig1_3.jpg"))
	plt.close()

def compute_entire_mse(image_list, path=test_path):
	vae_model = VAE_model(64, 64, 3)
	vae_model.load_weights(weight_path, by_name=True)

	entire_mse = 0
	n_entire_mse = 0
	for img_name in image_list:
		img = io.imread(os.path.join(path, img_name))
		input_img = normalize_image(img.reshape((1, ) + img.shape))
		result_img = vae_model.predict(input_img)
		output_result = process_image(result_img[0])
		
		input_flatten = img.flatten()
		output_flatten = output_result.flatten()
		entire_mse += mean_squared_error(input_flatten, output_flatten)

		n_input_flatten = input_img.flatten()
		n_output_flatten = result_img[0].flatten()
		n_entire_mse += mean_squared_error(n_input_flatten, n_output_flatten)

	average_mse = entire_mse / len(image_list)
	average_n_mse = n_entire_mse / len(image_list)

	return average_mse, average_n_mse

def random_generated_image():
	decoder_model = Decoder_model()
	decoder_model.load_weights(weight_path, by_name=True)

	# Random 32 latent vectors by seed(2)
	np.random.seed(2)
	rand_latent_vector = np.random.normal(size=(32, latent_dim))

	plt.figure(figsize=(16,8))
	for vector_index in range(rand_latent_vector.shape[0]):
		latent_vector = rand_latent_vector[vector_index].reshape(1, latent_dim)
		result_img = decoder_model.predict(latent_vector)
		output_result = process_image(result_img[0])
		plt.subplot(4, 8, vector_index + 1)
		plt.axis('off')
		plt.imshow(output_result)

	plt.savefig(os.path.join(output_path, "fig1_4.jpg"))
	plt.close()

def tSNE_plot_points(image_list, path=test_path):
	encoder_model = Encoeder_model(64, 64, 3)
	encoder_model.load_weights(weight_path, by_name=True)
	
	image = []
	color_label = []

	#Process image
	for img_name in image_list:
		img = io.imread(os.path.join(path, img_name))
		img = normalize_image(img)
		image.append(img)
	image = np.array(image)

	#Predict latent vector (Encoder)
	latent_vectors = encoder_model.predict(image)

	# Process label
	with open(label_path, 'r') as outfile:
		outfile.readline()
		for line in outfile:
			# Male or Female
			attribute = int(float(line.split(',')[8]))
			if attribute == 1:
				color_label.append('b')
			elif attribute == 0:
				color_label.append('r')

	# tSNE
	latent_vectors = latent_vectors.astype(np.float64)
	latent_embedded = TSNE(n_components=2, random_state=0).fit_transform(latent_vectors)
	point_X = latent_embedded[:,0]
	point_Y = latent_embedded[:,1]

	# Get info of point and plot point
	# label_1 = dict()
	# label_1['X'] = list()
	# label_1['Y'] = list()
	# label_0 = dict()
	# label_0['X'] = list()
	# label_0['Y'] = list()
	# for index in range(point_X.shape[0]):
	# 	if color_label[index] == 'b':
	# 		label_1['X'].append(point_X[index])
	# 		label_1['Y'].append(point_Y[index])
	# 	elif color_label[index] == 'r':
	# 		label_0['X'].append(point_X[index])
	# 		label_0['Y'].append(point_Y[index])

	plt.title("Result")
	plt.scatter(x=point_X, y=point_Y, c=color_label, s=[5])
	plt.text(40, -40, s="Male", fontdict={'size':16,'color':'b'})
	plt.text(-40, 40, s="Female", fontdict={'size':16,'color':'r'})
	plt.axis('off')

	plt.savefig(os.path.join(output_path, "fig1_5.jpg"))
	plt.close()

def read_image_name(path):
	file_list = [file for file in os.listdir(path) if file.endswith('.png')]
	file_list.sort()

	return file_list

def normalize_image(img):
	output_result = img / 127.5 - 1

	return output_result

def process_image(img):
	output_result = ((img + 1) * 127.5).astype(np.uint8)

	return output_result

def sampling(args):
	z_mean, z_log_var = args

	return z_mean

def Encoeder_model(input_height, input_width, input_channels):
	img_input = Input(shape=(input_height, input_width	, input_channels))
	# Encoder
	en_x = Conv2D(input_channels, kernel_size=(2, 2), padding='same', activation='relu', name='encoder_conv1')(img_input)
	en_x = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv2')(en_x)
	en_x = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv3')(en_x)
	en_x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv4')(en_x)
	en_x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv5')(en_x)
	en_x = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv6')(en_x)
	en_x = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv7')(en_x)

	# Latent vector
	z_mean = Conv2D(512, kernel_size=(1, 1), name='mean_layer')(en_x)
	z_mean = Reshape(target_shape=(512,), name='mean_reshape')(z_mean)

	model = Model(img_input, z_mean)

	return model

def Decoder_model():
	img_input = Input(shape=(512,))
	# Decoder
	z = Reshape(target_shape=(1, 1, 512), name='z_reshape')(img_input)
	de_x = Conv2DTranspose(256, kernel_size=(2, 2), padding='valid', activation='relu', name='decoder_deconv1')(z)
	de_x = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv2')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv3')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv4')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv5')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv6')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	# Use tanh (normalization : -1 ~ 1) sigmoid (normalization : 0 ~ 1)
	out = Conv2D(3, kernel_size=(1, 1), padding='valid', activation='tanh', name='decoder_mean_squash')(de_x)
	
	model = Model(img_input, out)

	return model

def VAE_model(input_height, input_width, input_channels):
	img_input = Input(shape=(input_height, input_width	, input_channels))
	# Encoder
	en_x = Conv2D(input_channels, kernel_size=(2, 2), padding='same', activation='relu', name='encoder_conv1')(img_input)
	en_x = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv2')(en_x)
	en_x = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv3')(en_x)
	en_x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv4')(en_x)
	en_x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv5')(en_x)
	en_x = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv6')(en_x)
	en_x = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv7')(en_x)

	# Latent vector
	z_mean = Conv2D(512, kernel_size=(1, 1), name='mean_layer')(en_x)
	z_mean = Reshape(target_shape=(512,), name='mean_reshape')(z_mean)
	z_log_var = Conv2D(512, kernel_size=(1, 1), name='logvar_layer')(en_x)
	z_log_var = Reshape(target_shape=(512,), name='logvar_reshape')(z_log_var)
	z = Lambda(sampling, output_shape=(latent_dim,), name='latent_layer')([z_mean, z_log_var])

	# Decoder
	z = Reshape(target_shape=(1, 1, 512), name='z_reshape')(z)
	de_x = Conv2DTranspose(256, kernel_size=(2, 2), padding='valid', activation='relu', name='decoder_deconv1')(z)
	de_x = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv2')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv3')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv4')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv5')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)
	de_x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv6')(de_x)
	de_x = Cropping2D(cropping=((1,1),(1,1)))(de_x)

	out = Conv2D(input_channels, kernel_size=(1, 1), padding='valid', activation='tanh', name='decoder_mean_squash')(de_x)
	
	model = Model(img_input, out)

	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', help='model path', type=str)
	parser.add_argument('-i', '--inputs', help='input dataset directory', type=str)
	parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output', type=str)
	args = parser.parse_args()

	weight_path = args.model
	test_path = os.path.join(args.inputs, 'test')
	label_path = os.path.join(args.inputs, 'test.csv')
	output_path = args.outputs

	print("----------------------- VAE processing ...... -----------------------")
	# Plot loss graph
	print("======================= Processing loss graph =======================")
	loss_graph()
	print("=======================       Finished        =======================")

	test_image_list = read_image_name(test_path)
	print("====================== Processing rconstructed ======================")
	reconstructed_imgage(test_image_list)
	mse, normal_mse = compute_entire_mse(test_image_list)
	print("MSE: {}\nNormalization_MSE: {}".format(mse, normal_mse))
	print("=======================       Finished        =======================")

	print("========================= Processing random =========================")
	random_generated_image()
	print("=======================       Finished        =======================")

	print("========================== Processing tSNE ==========================")
	tSNE_plot_points(test_image_list)
	print("=======================       Finished        =======================")
