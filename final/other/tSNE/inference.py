from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model, load_model
from keras import backend as K

import os, sys
import json
import numpy as np
import argparse
from time import gmtime, strftime
import scipy.misc
import matplotlib
from matplotlib import cm

matplotlib.use('Agg') # no window
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def imageNormalizer(img):
    img = img / 127.5 - 1
    return img

def load_matchTable():
    table = dict()
    pathTable = os.path.join(args.pathInfo, "tSNE_data.jsdb")
    with open(pathTable, 'r') as outfile:
        table = json.load(outfile)

    return table

def load_dataList(pathDataset):
    with open(pathDataset, 'r') as outfile:
        inputData = json.load(outfile)
	
    return list(inputData.keys()), list(inputData.values())

def tsne(features, label):
    numpy_features = np.array(features, dtype=np.float64)
    numpy_label = np.array(label)

    tsne = TSNE(n_components=2,
                learning_rate=200,
                perplexity=30,
                init='pca',
                n_iter=50000,
                verbose=1,
                random_state=2
                )

    latent_2d = tsne.fit_transform(numpy_features)
    latent_2d = np.array(latent_2d)

    plt.figure()
    latent_x = latent_2d[:, 0]
    latent_y = latent_2d[:, 1]
    
    color_map = {0:'slategray', 1:'r', 2:'g', 3:'y', 4:'c', 5:'m', 6:'b', 7:'peru', 8:'crimson', 9:'gold', 10:'indigo', 11:'royalblue'}
    colorList = [color_map[label] for label in numpy_label]
    plt.scatter(latent_x, latent_y, c=colorList, s=15)
    plt.axis('off')
    outfig_name = os.path.join(args.output_dir, "tSNE_{}.png".format(strftime("%Y%m%d_%H%M%S", gmtime())))
    plt.savefig(outfig_name)
    plt.close()

def main(args):
    matchTable = load_matchTable()
    
    mobile_model = load_model(args.pre_model)
    mobile_model.summary()
    
    base_model = Model(input=mobile_model.input, output=mobile_model.get_layer('global_average_pooling2d_1').output)

    validDataList, val_labels = load_dataList("./tSNE_data.jsdb")
    
    images = []
    for file in validDataList:
        filePath = os.path.join(args.data_dir, 'val', file)
        images.append(imageNormalizer(scipy.misc.imread(filePath)))
    images = np.array(images)
    val_features = base_model.predict(images)
    print(np.array(val_features).shape, np.array(val_labels).shape)
    tsne(val_features, val_labels)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference validation data')
    parser.add_argument('-m', '--pre_model', default=None, type=str, help='pretrained model')
    parser.add_argument('-p', '--pathInfo', default='./', type=str, help='facebank_information')
    parser.add_argument('-d', '--data_dir', default='./dlcv_final_2_dataset/', type=str, help='dataset directory')
    parser.add_argument('-o', '--output_dir', default='./', type=str, help='output directory')
    args = parser.parse_args()
    main(args)
