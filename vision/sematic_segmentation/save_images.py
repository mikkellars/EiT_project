"""
Script for saving image and masks.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

save_path = 'assets/data'

real_img = 'data/fence_data/train_set/images/2017_Train_00001.jpg'
real_mask = 'data/fence_data/train_set/labels/2017_Train_00001.png'
synth_img = 'synth_ml_data/cycles/0099.png'
synth_mask = 'synth_ml_data/object_index/0099.png'


def save_image(path: str, name: str):
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout(True)
    plt.savefig(f'{save_path}/{name}.png')


def save_mask(path: str, name: str):
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[img != 0] = 1
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout(True)
    plt.savefig(f'{save_path}/{name}.png')


save_image(real_img, 'real_image')
save_mask(real_mask, 'real_mask')

save_image(synth_img, 'synth_image')
save_mask(synth_mask, 'synth_mask')