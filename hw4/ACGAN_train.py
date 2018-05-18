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

class ACGAN(): 
	# Adam(0.0002, 0.5)
	def __init__(self, img_height, img_width, img_channels):
		self.img_shape = (img_height, img_width, img_channels)
		self.init = initializers.RandomNormal(mean=0.0, stddev=0.02)
		self.latent_dim = 128
		self.num_classes = 2
		self.d_rloss_label_his = []
		self.d_floss_label_his = []
		self.g_label_loss_his = []
		self.d_racc_his = []
		self.d_facc_his = []
		self.g_acc_his = []

		# Train discriminator => loss function = binary crossentropy
		self.discriminator = self.Discriminator_model()
		self.discriminator.compile(optimizer=Adam(0.0002, 0.5), 
							loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
							metrics=['accuracy'])

		# Train generator: only training the generator
		self.discriminator.trainable = False

		self.generator = self.Generator_model()

		z = Input(shape=(self.latent_dim,))
		label = Input(shape=(1,))
		gen_img = self.generator([z, label])

		valid, target_label = self.discriminator(gen_img)

		self.acgan = Model([z, label], [valid, target_label])
		self.acgan.compile(optimizer=Adam(0.0002, 0.5),
					loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
					metrics=['accuracy'])

	def Generator_model(self):
		noise = Input(shape=(self.latent_dim,), name='noise_input')

		# Add label as input
		label = Input(shape=(1,), name='label_input')

		model_input = Concatenate(axis=-1, name='model_input')([noise, label])

		gen_x = Dense(8*8*128, activation='relu')(model_input)
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

		model = Model([noise, label], img)

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

		dis_x = Dense(1, activation='sigmoid', name='discriminator_real')(flatten)
		label = Dense(self.num_classes+1, activation='softmax', name='discriminator_label')(flatten)

		model = Model(img, [dis_x, label])

		return model

	def read_image_name(self):
		file_list = [file for file in os.listdir(train_path) if file.endswith('.png')]
		file_list.sort()

		return file_list

	def read_label(self):
		label_list = []
		with open('./hw4_data/train.csv', 'r') as outfile:
			outfile.readline()
			for out_line in outfile:
				label = int(float(out_line.split(',')[10]))
				label_list.append(label)

		return np.array(label_list).astype(np.int32)

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
		noise = np.random.normal(0, 1, (10, self.latent_dim))

		plt.figure(figsize=(16,4))
		for noise_index in range(noise.shape[0]):
			# Create no_smile face images
			noise_vector = noise[noise_index].reshape(1, self.latent_dim)
			label = np.zeros((1, 1)).astype(np.int32)
			result_img = self.generator.predict([noise_vector, label])
			output_result = self.convert_image(result_img[0])
			plt.subplot(2, 10, noise_index + 1)
			plt.axis('off')
			plt.imshow(output_result)
			# Create smile face images
			label = np.ones((1, 1)).astype(np.int32)
			result_img = self.generator.predict([noise_vector, label])
			output_result = self.convert_image(result_img[0])
			plt.subplot(2, 10, noise_index + 11)
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

		label_list = self.read_label()

		# The ouptut of real images and fake images
		valid = np.ones((batch_size, 1))
		half_valid = np.ones((half_batch, 1))
		half_fake = np.zeros((half_batch, 1))

		# Data order
		data_order = np.arange(train_data_number)

		for epoch in range(epochs):
			print("Epoch: ", epoch)
			print("=============================================================================================================================")
			np.random.shuffle(data_order)
			for step in range(steps_per_epoch):
				# Real images
				batch_imgs = self.process_image(data_order[step * half_batch : (step+1) * half_batch], train_name)
				# Read labels of images
				batch_labels = label_list[data_order[step * half_batch : (step+1) * half_batch]]
				# Generate the images which is fake when training discriminator
				noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
				sample_labels = np.random.randint(0, self.num_classes, (half_batch, 1))
				gen_imgs = self.generator.predict([noise, sample_labels])

				#-----------------------------------------------#
				# Train discriminator with real and fake images #
				#-----------------------------------------------#
				fake_labels = 2 * np.ones((half_batch, 1))
				for layer in self.discriminator.layers:
					layer.trainable = True
				d_loss_real = self.discriminator.train_on_batch(batch_imgs, [half_valid, batch_labels])
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [half_fake, fake_labels])
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				#---------------------#
				# Train generator     #
				#---------------------#
				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
				sample_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
				for layer in self.discriminator.layers:
					layer.trainable = False
				g_loss = self.acgan.train_on_batch([noise, sample_labels], [valid, sample_labels])

				if(step % 100 == 0):
					print('Step {} :'.format(step))
					print("Discriminator model: Loss (Real or Fake) = {} , Real loss = {} , Fake loss = {}".format(d_loss[1], d_loss_real[1], d_loss_fake[1]))
					print("Discriminator model: Accuray (Real or Fake) = {} , Real accuray = {} , Fake accuray = {}".format(d_loss[3], d_loss_real[3], d_loss_fake[3]))
					print("ACGAN model: Loss (Real or Fake) = {}".format(g_loss[1]))
					print("ACGAN model: Accuray (Real or Fake) = {}".format(g_loss[3]))
					print("-----------------------------------------------------------------------------------------------------------------------------")
					print("Discriminator model: Loss (Label) = {} , Real loss = {} , Fake loss = {}".format(d_loss[2], d_loss_real[2], d_loss_fake[2]))
					print("Discriminator model: Accuray (Label) = {} , Real accuray = {} , Fake accuray = {}".format(d_loss[4], d_loss_real[4], d_loss_fake[4]))
					print("ACGAN model: Loss (Label) = {}".format(g_loss[2]))
					print("ACGAN model: Accuray (Label) = {}".format(g_loss[4]))
					print("=============================================================================================================================")
					# Record loss
					self.d_rloss_label_his.append(d_loss_real[2])
					self.d_floss_label_his.append(d_loss_fake[2])
					self.g_label_loss_his.append(g_loss[2])
					self.d_racc_his.append(d_loss_real[3])
					self.d_facc_his.append(d_loss_fake[3])
					self.g_acc_his.append(g_loss[3])

			self.test_process_image(epoch)
			self.discriminator.save("./model/discriminator_{}.h5".format(epoch))
			self.generator.save("./model/generator_{}.h5".format(epoch))

def loss_graph():
	with open('ACGAN_loss_history.pickle', 'rb') as outfile:
		loss_history = pickle.load(outfile)

	plt.figure(figsize=(16,6))

	plt.subplot(1, 2, 1)
	real_line, = plt.plot(loss_history['Discriminator_real_loss'], linewidth=1, color='red', label='real_loss')
	fake_line, = plt.plot(loss_history['Discriminator_fake_loss'], linewidth=1, color='blue', label='blue_loss')
	plt.legend(handles=[real_line, fake_line])
	plt.xlabel("Training Steps (100 steps)")
	plt.title("Discriminator Loss")

	plt.subplot(1, 2, 2)
	plt.plot(loss_history['Generator_loss'], linewidth=1)
	plt.xlabel("Training Steps (100 steps)")
	plt.title("Generator Loss")

	plt.savefig("fig2_2.jpg")
	plt.close()

if __name__ == '__main__':
	acgan = ACGAN(64, 64, 3)
	acgan.acgan.summary()
	acgan.generator.summary()
	acgan.discriminator.summary()
	acgan.train(200, 20)

	loss_history = dict()
	loss_history['Discriminator_real_label_loss'] = acgan.d_rloss_label_his
	loss_history['Discriminator_fake_label_loss'] = acgan.d_floss_label_his
	loss_history['Generator_label_loss'] = acgan.g_label_loss_his
	loss_history['Discriminator_real_accuracy'] = acgan.d_racc_his
	loss_history['Discriminator_fake_accuracy'] = acgan.d_facc_his
	loss_history['Generator_accuracy'] = acgan.g_acc_his

	with open('ACGAN_loss_history.pickle', 'wb') as outfile:
		pickle.dump(loss_history, outfile)
	# loss_graph()
	# gan.gan.summary()
	# gan.discriminator.summary()
	# gan.generator.summary()
	
