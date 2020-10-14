"""
Trainer of the Faster R-CNN.
"""


import os
import sys
sys.path.append(os.getcwd())


import torch
import matplotlib
from tqdm import tqdm
from torch.utils import data
from vision.faster_rcnn.helpers import to_numpy, to_tensor, scalar
from vision.faster_rcnn.model import FasterRCNNVGG16


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser('Faster R-CNN trainer')
    parser.add_argument('--data_dir', type=str, default='', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='', help='path to save directory')
    parser.add_argument('--model_dir', type=str, default='', help='path to models directory')
    parser.add_argument('--resume_model', type=str, default='', help='path to model to resume')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to run')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    return args


def main(args):
    


if __name__ == '__main__':
    args = parse_arguments()
    main(args)