"""
This module create new image of a dataset by creating patches of the images in a size.
"""


import os
import sys
sys.path.append(os.getcwd())

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from split_data import split_data


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Create patches of dataset')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/test_set', help='path to image directory')
    parser.add_argument('--save_dir', type=str, default='vision/data/fence_data/patch_test_set', help='path to save directory')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size of the new image')
    parser.add_argument('--split_test_val', action='store_true', help='create 2 txt files with training and validation data in each')
    args = parser.parse_args()
    return args


def main(args):
    start_time = time.time()

    img_dir = args.data_dir + '/images'
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    images = sorted(images)

    mask_dir = args.data_dir + '/labels'
    masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    masks = sorted(masks)

    loop = tqdm(range(len(images)))
    for idx in loop:
        img = cv2.imread(images[idx])
        mask = cv2.imread(masks[idx])

        h, w, _ = img.shape
        rows = h // args.patch_size
        cols = w // args.patch_size

        for i in range(0, rows):
            for j in range(0, cols):
                ymin = i * h // rows
                ymax = i * h // rows + h // rows
                xmin = j * w // cols
                xmax = j * w // cols + w // cols

                roi_img = img[ymin : ymax, xmin : xmax]
                roi_mask = mask[ymin : ymax, xmin : xmax]
                
                roi_img = cv2.resize(roi_img, (args.patch_size, args.patch_size), interpolation=cv2.INTER_CUBIC)
                roi_mask = cv2.resize(roi_mask, (args.patch_size, args.patch_size), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(f'{args.save_dir}/images/{idx:05d}.png', roi_img)
                cv2.imwrite(f'{args.save_dir}/labels/{idx:05d}.png', roi_mask)
    
    if args.split_train_val:
        split_data(f'{args.save_dir}/images', args.save_dir, True)

    end_time = time.time() - start_time
    print(f'Done! It took {end_time:.04f} seconds')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
