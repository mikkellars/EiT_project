"""Script for converting images.
"""


import os
import cv2
import time
import argparse
import numpy as np


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser('Conversion of images.')
    parser.add_argument('--folder', type=str, default='/home/mathias/Documents/experts_in_teams_proj/vision/data/real_data/raw', help='path to folder')
    args = parser.parse_args()

    files = [os.path.join(args.folder, f) for f in os.listdir(args.folder)]

    for i, f in enumerate(files):
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow('Input', im)
        
        key = cv2.waitKey()
        if key == ord('q'):
            break

    end_time = time.time() - start_time
    print(f'It took {end_time//60:0.0f} minutes and {end_time%60:0.0f} seconds.')
    print('Done!')
