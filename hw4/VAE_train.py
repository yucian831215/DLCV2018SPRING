import os
import keras
import pickle

import numpy as np

from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from skimage import io

from keras import backend as K

train_path = './hw4_data/train/'

intermediate_dim = 128
latent_dim = 512
KL_weight = 1e-4

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.mse_losses = []
		self.kl_losses = []

	def on_batch_end(self, batch, logs={}):
		self.mse_losses.append(logs.get('decoder_mean_squash_loss', 0))
		self.kl_losses.append(logs.get('KL_loss_loss', 0))

def read_image_name(path):
	file_list = [file for file in os.listdir(path) if file.endswith('.png')]
	file_list.sort()

	return file_list

def load_train_batch_data(file_name, batch_size, data_size, path=train_path):
	start_index = 0
	while True:
		X = []
		for img_index in range(batch_size):
			img_X = io.imread(os.path.join(path, file_name[start_index + img_index]))
			img_X = img_X / 127.5 - 1
			X.append(img_X)

		X = np.array(X)
		Y = [X, np.random.rand(X.shape[0], 1)]
		yield(X, Y)

		start_index += batch_size
		if start_index >= data_size:
			start_index = 0	

def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)

	return z_mean + K.exp(z_log_var) * epsilon

def VAE_model(input_height, input_width, input_channels):
	img_input = Input(shape=(input_height, input_width	, input_channels))
	# Encoder
	en_x = Conv2D(input_channels, kernel_size=(2, 2), padding='same', activation='relu', name='encoder_conv1')(img_input)
	en_x = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv2')(en_x)
	en_x = BatchNormalization()(en_x)
	en_x = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv3')(en_x)
	en_x = BatchNormalization()(en_x)
	en_x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv4')(en_x)
	en_x = BatchNormalization()(en_x)
	en_x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv5')(en_x)
	en_x = BatchNormalization()(en_x)
	en_x = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv6')(en_x)
	en_x = BatchNormalization()(en_x)
	en_x = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu', name='encoder_conv7')(en_x)
	en_x = BatchNormalization()(en_x)

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
	kl_loss = Lambda(lambda x: (-0.5) * K.sum(1 + x[1] - K.square(x[0]) - K.exp(x[1]), axis=-1, keepdims=True),
					name='KL_loss')([z_mean, z_log_var])

	model = Model(img_input, output=[out, kl_loss])

	model.compile(optimizer=Adam(lr=1e-4), loss=["mean_squared_error", lambda y_true,y_pred: y_pred], loss_weights=[1, KL_weight])

	return model

if __name__ == '__main__':

	epochs = 200
	batch_size = 8

	model = VAE_model(64, 64, 3)

	model.summary()

	checkpoint = ModelCheckpoint('vae_model.h5', monitor='loss', save_best_only=True)

	train_name = read_image_name(train_path)
	train_data_number = len(train_name)
	train_batch_number = int(train_data_number / batch_size)

	train_generator = load_train_batch_data(train_name, batch_size, train_data_number)

	history = LossHistory()

	model.fit_generator(train_generator,
				steps_per_epoch=train_batch_number,
				epochs=epochs,
				callbacks=[history, checkpoint])

	# Record losses
	loss_history = dict()
	loss_history['MSE'] = history.mse_losses
	loss_history['KL'] = history.kl_losses

	with open('VAE_loss_history.pickle', 'wb') as outfile:
		pickle.dump(loss_history, outfile)
