#!/bin/bash
python3 VAE_test.py -m './vae_model.h5' -i $1 -o $2
python3 GAN_test.py -m './gan_generator.h5' -o $2
python3 ACGAN_test.py -m './acgan_generator.h5' -o $2