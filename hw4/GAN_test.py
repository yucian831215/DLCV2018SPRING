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

generator_path = './gan_generator.h5'
loss_path = './GAN_loss_history.pickle'
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
	Discriminator_real_loss_X = np.arange(len(loss_history['Discriminator_real_loss']))
	Discriminator_real_loss_Y = np.array(loss_history['Discriminator_real_loss'])
	smooth = InterpolatedUnivariateSpline(Discriminator_real_loss_X, Discriminator_real_loss_Y)
	Discriminator_real_loss_X_smooth = np.linspace(Discriminator_real_loss_X.min(), Discriminator_real_loss_X.max(), num=500).astype(np.int32)
	Discriminator_real_loss_Y_smooth = smooth(Discriminator_real_loss_X_smooth)

	plt.plot(Discriminator_real_loss_X, Discriminator_real_loss_Y, linewidth=1, color='mistyrose')

	Discriminator_fake_loss_X = np.arange(len(loss_history['Discriminator_fake_loss']))
	Discriminator_fake_loss_Y = np.array(loss_history['Discriminator_fake_loss'])
	smooth = InterpolatedUnivariateSpline(Discriminator_fake_loss_X, Discriminator_fake_loss_Y)
	Discriminator_fake_loss_X_smooth = np.linspace(Discriminator_fake_loss_X.min(), Discriminator_fake_loss_X.max(), num=500).astype(np.int32)
	Discriminator_fake_loss_Y_smooth = smooth(Discriminator_fake_loss_X_smooth)

	plt.plot(Discriminator_fake_loss_X, Discriminator_fake_loss_Y, linewidth=1, color='lavender')

	real_line, = plt.plot(Discriminator_real_loss_X_smooth, Discriminator_real_loss_Y_smooth, linewidth=1.5, color='red', label='real_loss')
	fake_line, = plt.plot(Discriminator_fake_loss_X_smooth, Discriminator_fake_loss_Y_smooth, linewidth=1.5, color='blue', label='fake_loss')

	plt.legend(handles=[real_line, fake_line])
	plt.xlabel("Training Steps (100 steps)")
	plt.title("Discriminator Loss")

	plt.subplot(1, 2, 2)
	Generator_loss_X = np.arange(len(loss_history['Generator_loss']))
	Generator_loss_Y = np.array(loss_history['Generator_loss'])
	smooth = InterpolatedUnivariateSpline(Generator_loss_X, Generator_loss_Y)
	Generator_loss_X_smooth = np.linspace(Generator_loss_X.min(), Generator_loss_X.max(), num=500).astype(np.int32)
	Generator_loss_Y_smooth = smooth(Generator_loss_X_smooth)

	plt.plot(Generator_loss_X, Generator_loss_Y, linewidth=1, color='lavender')
	plt.plot(Generator_loss_X_smooth, Generator_loss_Y_smooth, linewidth=1.5, color='blue', label='generator_loss')
	plt.xlabel("Training Steps (100 steps)")
	plt.title("Generator Loss")

	plt.savefig(os.path.join(output_path, "fig2_2.jpg"))
	plt.close()

def test_process_image():
	generator = load_model(generator_path)
	
	np.random.seed(3)
	noise = np.random.normal(0, 1, (32, latent_dim))

	plt.figure(figsize=(16,8))
	for noise_index in range(noise.shape[0]):
		noise_vector = noise[noise_index].reshape(1, latent_dim)
		result_img = generator.predict(noise_vector)
		output_result = convert_image(result_img[0])
		plt.subplot(4, 8, noise_index + 1)
		plt.axis('off')
		plt.imshow(output_result)

	plt.savefig(os.path.join(output_path, "fig2_3.jpg"))
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', help='model path', type=str)
	parser.add_argument('-o', '--outputs', help='output dataset directory', default='./output', type=str)
	args = parser.parse_args()

	generator_path = args.model
	output_path = args.outputs

	print("----------------------- GAN processing ...... -----------------------")

	print("======================= Processing loss graph =======================")
	loss_graph()
	print("=======================       Finished        =======================")

	print("========================= Processing random =========================")
	test_process_image()
	print("=======================       Finished        =======================")