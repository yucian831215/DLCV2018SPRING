import os
import keras
import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from skimage import io

from keras import backend as K

train_path = './hw4_data/train/'

class GAN():
	# Adam(0.0002, 0.5)
	def __init__(self, img_height, img_width, img_channels):
		self.img_shape = (img_height, img_width, img_channels)
		self.init = initializers.RandomNormal(mean=0.0, stddev=0.02)
		self.latent_dim = 128
		self.d_rloss_his = []
		self.d_floss_his = []
		self.g_loss_his = []

		# Train discriminator => loss function = binary crossentropy
		self.discriminator = self.Discriminator_model()
		self.discriminator.compile(optimizer=Adam(0.0002, 0.5), 
							loss='binary_crossentropy',
							metrics=['accuracy'])

		# Train generator: only training the generator
		self.discriminator.trainable = False

		self.generator = self.Generator_model()

		z = Input(shape=(self.latent_dim,))
		gen_img = self.generator(z)

		valid = self.discriminator(gen_img)

		self.gan = Model(z, valid)
		self.gan.compile(optimizer=Adam(0.0002, 0.5),
					loss='binary_crossentropy',
					metrics=['accuracy'])

	def Generator_model(self):
		noise = Input(shape=(self.latent_dim,))

		gen_x = Dense(8*8*128, activation='relu')(noise)
		gen_x = BatchNormalization(momentum=0.8)(gen_x)

		gen_x = Reshape((8,8,128))(gen_x)
		gen_x = UpSampling2D()(gen_x)
		gen_x = Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
							padding='same', kernel_initializer=self.init, name='generator_conv1')(gen_x)
		gen_x = BatchNormalization(momentum=0.8)(gen_x)

		gen_x = UpSampling2D()(gen_x)
		gen_x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
							padding='same', kernel_initializer=self.init, name='generator_conv2')(gen_x)
		gen_x = BatchNormalization(momentum=0.8)(gen_x)

		gen_x = UpSampling2D()(gen_x)
		gen_x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
							padding='same', kernel_initializer=self.init, name='generator_conv3')(gen_x)
		gen_x = BatchNormalization(momentum=0.8)(gen_x)

		img = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), activation='tanh', 
							padding='same', kernel_initializer=self.init, name='generator_conv4')(gen_x)

		model = Model(noise, img)

		return model

	def Discriminator_model(self):
		img = Input(shape=self.img_shape)

		dis_x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
							kernel_initializer=self.init, name='discriminator_conv1')(img)
		dis_x = LeakyReLU(alpha=0.2)(dis_x)
		dis_x = BatchNormalization(momentum=0.8)(dis_x)
		dis_x = Dropout(0.25)(dis_x)

		dis_x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
							kernel_initializer=self.init, name='discriminator_conv2')(dis_x)
		dis_x = LeakyReLU(alpha=0.2)(dis_x)
		dis_x = BatchNormalization(momentum=0.8)(dis_x)
		dis_x = Dropout(0.25)(dis_x)

		dis_x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', 
							kernel_initializer=self.init, name='discriminator_conv3')(dis_x)
		dis_x = LeakyReLU(alpha=0.2)(dis_x)
		dis_x = BatchNormalization(momentum=0.8)(dis_x)
		dis_x = Dropout(0.25)(dis_x)

		dis_x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', 
							kernel_initializer=self.init, name='discriminator_conv4')(dis_x)
		dis_x = LeakyReLU(alpha=0.2)(dis_x)
		dis_x = BatchNormalization(momentum=0.8)(dis_x)
		dis_x = Dropout(0.25)(dis_x)

		flatten = Flatten()(dis_x)

		dis_x = Dense(1, activation='sigmoid', name='discriminator_fc1')(flatten)

		model = Model(img, dis_x)

		return model

	def read_image_name(self):
		file_list = [file for file in os.listdir(train_path) if file.endswith('.png')]
		file_list.sort()

		return file_list

	def process_image(self, img_index, file_name):
		real_imgs = []
		for i in img_index:
			real_img = io.imread(os.path.join(train_path, file_name[i]))
			real_img = (real_img.astype(np.float32) - 127.5) / 127.5
			real_imgs.append(real_img)

		return np.array(real_imgs)

	def convert_image(self, img):
		output_result = (img * 127.5 + 127.5).astype(np.uint8)

		return output_result

	def test_process_image(self, epoch):
		np.random.seed(2)
		noise = np.random.normal(0, 1, (32, self.latent_dim))

		plt.figure(figsize=(16,8))
		for noise_index in range(noise.shape[0]):
			noise_vector = noise[noise_index].reshape(1, self.latent_dim)
			result_img = self.generator.predict(noise_vector)
			output_result = self.convert_image(result_img[0])
			plt.subplot(4, 8, noise_index + 1)
			plt.axis('off')
			plt.imshow(output_result)

		plt.savefig("./Output/Epoch_{}_random.jpg".format(epoch))
		plt.close()
		# Recover seed
		np.random.seed(None)

	def train(self, epochs, batch_size):
		train_name = self.read_image_name()
		train_data_number = len(train_name)
		half_batch = int(batch_size / 2)
		steps_per_epoch = int(train_data_number / half_batch)

		# The ouptut of real images and fake images
		valid = np.ones((batch_size, 1))
		half_valid = np.ones((half_batch, 1))
		half_fake = np.zeros((half_batch, 1))

		# Data order
		data_order = np.arange(train_data_number)

		for epoch in range(epochs):
			print("Epoch: ", epoch)
			np.random.shuffle(data_order)
			for step in range(steps_per_epoch):
				# Real images
				batch_imgs = self.process_image(data_order[step * half_batch : (step+1) * half_batch], train_name)
				# Generate the images which is fake when training discriminator
				noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
				gen_imgs = self.generator.predict(noise)

				#-----------------------------------------------#
				# Train discriminator with real and fake images #
				#-----------------------------------------------#
				for layer in self.discriminator.layers:
					layer.trainable = True
				d_loss_real = self.discriminator.train_on_batch(batch_imgs, half_valid)
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, half_fake)
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				#---------------------#
				# Train generator     #
				#---------------------#
				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
				for layer in self.discriminator.layers:
					layer.trainable = False
				g_loss = self.gan.train_on_batch(noise, valid)

				if(step % 100 == 0):
					print('Step {} :'.format(step))
					print("Discriminator model: Loss = {} , Real loss = {} , Fake loss = {}".format(d_loss[0], d_loss_real[0], d_loss_fake[0]))
					print("Discriminator model: Accuray = {} , Real accuray = {} , Fake accuray = {}".format(d_loss[1], d_loss_real[1], d_loss_fake[1]))
					print("GAN model: Loss = {}".format(g_loss[0]))
					print("GAN model: Accuray = {}".format(g_loss[1]))
					# Record loss
					self.d_rloss_his.append(d_loss_real[0])
					self.d_floss_his.append(d_loss_fake[0])
					self.g_loss_his.append(g_loss[0])

			self.test_process_image(epoch)
			self.discriminator.save("./model/discriminator_{}.h5".format(epoch))
			self.generator.save("./model/generator_{}.h5".format(epoch))

if __name__ == '__main__':
	gan = GAN(64, 64, 3)
	gan.gan.summary()
	gan.generator.summary()
	gan.discriminator.summary()
	gan.train(200, 20)

	loss_history = dict()
	loss_history['Discriminator_real_loss'] = gan.d_rloss_his
	loss_history['Discriminator_fake_loss'] = gan.d_floss_his
	loss_history['Generator_loss'] = gan.g_loss_his

	with open('GAN_loss_history.pickle', 'wb') as outfile:
		pickle.dump(loss_history, outfile)

	loss_graph()
	
