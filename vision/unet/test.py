"""
"""


import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
from model import *
from utils.utils import * 


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Testing of SegNet for fence detection')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/test_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='vision/unet/images', help='path to directory to save figures')
    parser.add_argument('--model', type=str, default='vision/unet/models/detectnet_model.pt', help='path to model file')
    parser.add_argument('--show', action='store_false', help='show images and segmentation')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for fetching data')
    args = parser.parse_args()
    return args


def show(img, mask, path):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.title('Mask')
    plt.axis('off')
    plt.tight_layout(True)
    # plt.show()
    plt.savefig(path)
    plt.close('all')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = DetectNetDataset(args.data_dir, 'test', False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    saved_model = torch.load(args.model)
    print(f'Loaded {args.model}. The model is trained for {saved_model["epoch"]} epochs with {saved_model["loss"]} loss')

    model = UNet(15, 1)
    model.load_state_dict(saved_model["model_state_dict"])
    model.to(device)

    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    accs, intersections, unions, ious, times = [], [], [], [], []

    loop = tqdm(dataloader)
    start_time = time.time()
    for i, data in enumerate(loop):
        model.eval()

        img, mask = data

        img = img.to(device, dtype=torch.float32)
        mask_type = torch.float32 if model.n_classes == 1 else torch.long
        mask = mask.to(device, dtype=mask_type)
        mask = mask.view(mask.size(0), 1, mask.size(1), mask.size(2))

        with torch.set_grad_enabled(False):
            pred_time = time.time()
            pred = model(img)
            times.append((time.time() - pred_time))

            loss = criterion(pred, mask)

        pred = pred.data.cpu().detach().numpy()[0]
        pred[pred > 0] = 1
        pred[pred < 0] = 0

        mask = mask.data.cpu().detach().numpy()[0]

        acc, pix = accuracy(pred, mask)
        accs.append(acc)

        intersection, union = intersection_over_union(pred, mask, n_classes)
        intersections.append(intersection)
        unions.append(union)
        ious.append(intersection / union)

        loop.set_description(f'[Batch {i+1:03d}/{len(dataloader):03d}] [Loss {loss.item():.6f}] [Accuracy {np.mean(accs):.4f}] [IoU {np.mean(ious):.4f}] [Prediction time {np.mean(times):.4f}]')

        if args.show:
            img = img[0].data.cpu().detach().numpy().astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            show(img[:, :, :3], pred[0]*255, f'{args.save_dir}/{i+1:04d}.png')

    for i, iou in enumerate(np.transpose(ious)):
        print(f'{classes[i]} => IoU: mean = {np.mean(iou):.4f} (+/ {np.std(iou):4f})')

    print(f'IoU: mean = {np.mean(ious):.4f} (+/- {np.std(ious):.4f})')
    print(f'Accuracy: mean = {np.mean(accs):.4f} (+/- {np.std(accs):.4f})')
    print(f'Prediction time: mean = {np.mean(times):.4f} (+/- {np.std(times):.4f})')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
