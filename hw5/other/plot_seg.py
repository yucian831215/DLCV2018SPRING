import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import math

predict = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        predict.append(int(line.split('\n')[0]))

true_label = []
with open(sys.argv[2], 'r') as f:
    for line in f:
        true_label.append(int(line.split('\n')[0]))

img_dir = "./plot_image"
img_list = [name for name in os.listdir(img_dir)]
img_list.sort()

color_dict = {0:'darkgray', 1:'orange', 2:'r', 3:'b', 4:'y', 5:'m', 6:'deepskyblue', 7:'saddlebrown', 8:'sage', 9:'orchid', 10:'indigo'}

f, ax = plt.subplots(4, 2, figsize=(18, 3), gridspec_kw = {'height_ratios':[1, 1, 3, 1], 'width_ratios':[1, 12]})
f.tight_layout(h_pad=-2, w_pad=-2)

for i in range(408, 708):
    left = i-1
    if i == 0:
        left = 0
    ax[1, 1].barh(0, 5, left=left, color=color_dict[true_label[i]])
    ax[3, 1].barh(0, 5, left=left, color=color_dict[predict[i]])


for i in range(4):
    for j in range(2):
        ax[i, j].axis('off')

step_num = math.floor(len(img_list) / 11)
for i in range(11):
    x = 0.105 + i * 0.0763
    ax_img = f.add_axes([x, 0.264, 0.0763, 0.4])
    img = plt.imread(os.path.join(img_dir, img_list[i * step_num]))
    ax_img.imshow(img)
    ax_img.axis('off')

ax[1, 0].text(0.04, 0.4, "GroundTruth", fontsize=14)
ax[2, 0].text(0.17, 0.47, "Frames", fontsize=14)
ax[3, 0].text(0.1, 0.4, "Prediction", fontsize=14)

ax[0, 1].text(0.002, 0.3, "Other", fontsize=12)
ax[0, 1].text(0.057, 0.3, "Take", fontsize=12)
ax[0, 1].text(0.225, 0.3, "Other", fontsize=12)
ax[0, 1].text(0.45, 0.3, "Inspect/Read", fontsize=12)
ax[0, 1].text(0.615, 0.3, "Open", fontsize=12)
ax[0, 1].text(0.69, 0.3, "Divide/Pull Apart", fontsize=12)

plt.savefig("fig3_3.jpg")
plt.close()









