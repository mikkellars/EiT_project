

import os
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
from utils import *


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Testing of SegNet for fence detection')
    parser.add_argument('--data_dir', type=str, default='data/fence_data/test_set', help='path to data directory')
    parser.add_argument('--model', type=str, default='models/model.pt', help='path to model file')
    parser.add_argument('--show', action='store_false', help='show images and segmentation')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for fetching data')
    args = parser.parse_args()
    return args


def show(img, mask):
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
    plt.show()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = FenceDataset(args.data_dir, 'test', False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    saved_model = torch.load(args.model)
    print(f'Loaded {args.model}. The model is trained for {saved_model["epoch"]} epochs with {saved_model["loss"]} loss')

    model = Segnet(3, n_classes)
    model.load_state_dict(saved_model["model_state_dict"])
    model.to(device)

    criterion = Loss()

    accs, intersections, unions, ious, times = [], [], [], [], []

    loop = tqdm(dataloader)
    start_time = time.time()
    for i, data in enumerate(loop):
        model.eval()

        img, mask = data

        img = Variable(img).to(device)
        mask = Variable(mask).to(device)

        with torch.set_grad_enabled(False):
            pred_time = time.time()
            pred = model(img)
            times.append((time.time() - pred_time))

            loss = criterion(pred, mask)

        pred = pred.data.cpu().detach().numpy()[0][1]
        pred[pred > 0] = 1
        pred[pred < 0] = 0

        mask = mask.data.cpu().detach().numpy()

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
            show(img, pred*255)

    for i, iou in enumerate(np.transpose(ious)):
        print(f'{classes[i]} => IoU: mean = {np.mean(iou):.4f} (+/ {np.std(iou):4f})')

    print(f'IoU: mean = {np.mean(ious):.4f} (+/- {np.std(ious):.4f})')
    print(f'Accuracy: mean = {np.mean(accs):.4f} (+/- {np.std(accs):.4f})')
    print(f'Prediction time: mean = {np.mean(times):.4f} (+/- {np.std(times):.4f})')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
