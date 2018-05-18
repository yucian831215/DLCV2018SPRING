import os
import pickle
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from skimage import io
from sklearn.manifold import TSNE
from scipy.interpolate import InterpolatedUnivariateSpline

generator_path = './acgan_generator.h5'
loss_path = './ACGAN_loss_history.pickle'
output_path = './'
latent_dim = 128

def convert_image(img):
	output_result = (img * 127.5 + 127.5).astype(np.uint8)

	return output_result

def loss_graph():
	with open(loss_path, 'rb') as outfile:
		loss_history = pickle.load(outfile)

	plt.figure(figsize=(16,6))

	plt.subplot(1, 2, 1)
	Discriminator_real_label_loss_X = np.arange(len(loss_history['Discriminator_real_label_loss']))
	Discriminator_real_label_loss_Y = np.array(loss_history['Discriminator_real_label_loss'])
	smooth = InterpolatedUnivariateSpline(Discriminator_real_label_loss_X, Discriminator_real_label_loss_Y)
	Discriminator_real_label_loss_X_smooth = np.linspace(Discriminator_real_label_loss_X.min(), Discriminator_real_label_loss_X.max(), num=500).astype(np.int32)
	Discriminator_real_label_loss_Y_smooth = smooth(Discriminator_real_label_loss_X_smooth)

	plt.plot(Discriminator_real_label_loss_X, Discriminator_real_label_loss_Y, linewidth=1, color='mistyrose')

	Discriminator_fake_label_loss_X = np.arange(len(loss_history['Discriminator_fake_label_loss']))
	Discriminator_fake_label_loss_Y = np.array(loss_history['Discriminator_fake_label_loss'])
	smooth = InterpolatedUnivariateSpline(Discriminator_fake_label_loss_X, Discriminator_fake_label_loss_Y)
	Discriminator_fake_label_loss_X_smooth = np.linspace(Discriminator_fake_label_loss_X.min(), Discriminator_fake_label_loss_X.max(), num=500).astype(np.int32)
	Discriminator_fake_label_loss_Y_smooth = smooth(Discriminator_fake_label_loss_X_smooth)

	plt.plot(Discriminator_fake_label_loss_X, Discriminator_fake_label_loss_Y, linewidth=1, color='lavender')

	real_line, = plt.plot(Discriminator_real_label_loss_X_smooth, Discriminator_real_label_loss_Y_smooth, linewidth=1.5, color='red', label='real_loss')
	fake_line, = plt.plot(Discriminator_fake_label_loss_X_smooth, Discriminator_fake_label_loss_Y_smooth, linewidth=1.5, color='blue', label='fake_loss')
	
	plt.legend(handles=[real_line, fake_line], loc=1)
	plt.xlabel("Training Steps (100 steps)")
	plt.title("Training Loss of Attribute Classification (Discriminator)")

	plt.subplot(1, 2, 2)
	Generator_label_loss_X = np.arange(len(loss_history['Generator_label_loss']))
	Generator_label_loss_Y = np.array(loss_history['Generator_label_loss'])
	smooth = InterpolatedUnivariateSpline(Generator_label_loss_X, Generator_label_loss_Y)
	Generator_label_loss_X_smooth = np.linspace(Generator_label_loss_X.min(), Generator_label_loss_X.max(), num=500).astype(np.int32)
	Generator_label_loss_Y_smooth = smooth(Generator_label_loss_X_smooth)

	plt.plot(Generator_label_loss_X, Generator_label_loss_Y, linewidth=1, color='honeydew')
	
	generator_line, = plt.plot(Generator_label_loss_X_smooth, Generator_label_loss_Y_smooth, linewidth=1.5, color='green', label='generator_loss')

	plt.legend(handles=[generator_line], loc=1)
	plt.xlabel("Training Steps (100 steps)")
	plt.title("Training Loss of Attribute Classification (Generator)")

	plt.savefig(os.path.join(output_path, "fig3_2.jpg"))
	plt.close()

def test_process_image():
	generator = load_model(generator_path)
	
	np.random.seed(23)
	noise = np.random.normal(0, 1, (10, latent_dim))

	plt.figure(figsize=(16,4))
	plt.subplot(2, 11, 1)
	plt.axis('off')
	plt.text(-0.2, 0.5, "No Smiling", size=16)

	plt.subplot(2, 11, 12)
	plt.axis('off')
	plt.text(0, 0.5, "Smiling", size=16)

	for noise_index in range(noise.shape[0]):
		# Create no_smile face images
		noise_vector = noise[noise_index].reshape(1, latent_dim)
		label = np.zeros((1, 1)).astype(np.int32)
		result_img = generator.predict([noise_vector, label])
		output_result = convert_image(result_img[0])
		plt.subplot(2, 11, noise_index + 2)
		plt.axis('off')
		plt.imshow(output_result)
		# Create smile face images
		label = np.ones((1, 1)).astype(np.int32)
		result_img = generator.predict([noise_vector, label])
		output_result = convert_image(result_img[0])
		plt.subplot(2, 11, noise_index + 13)
		plt.axis('off')
		plt.imshow(output_result)

	plt.savefig(os.path.join(output_path, "fig3_3.jpg"))
	plt.close()

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-m', '--model', help='model path', type=str)
	# parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output', type=str)
	# args = parser.parse_args()

	# generator_path = args.model
	# output_path = args.outputs

	# print("---------------------- ACGAN processing ...... ----------------------")

	print("======================= Processing loss graph =======================")
	loss_graph()
	print("=======================       Finished        =======================")

	print("========================= Processing random =========================")
	test_process_image()
	print("=======================       Finished        =======================")