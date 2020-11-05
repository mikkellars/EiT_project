"""
Corner detection for fence inspection
"""


import os
import sys
sys.path.append(os.getcwd())

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Corner detection for fence inspection')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/test_set/labels', help='path to data directory')
    args = parser.parse_args()
    return args


def main(args):
    
    files = [f'{args.data_dir}/{f}' for f in os.listdir(args.data_dir)]
    for f in files:
        img = cv2.imread(f, 0)

        kernel = np.ones((10,10), np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)

        plt.imshow(img, cmap='gray'), plt.axis('off'), plt.show()

        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
        img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')
