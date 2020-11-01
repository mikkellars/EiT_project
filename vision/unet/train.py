"""
Train UNet.
"""


import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import fastai.vision as faiv
from dataset import *
from model import *
# from vision.unet.dataset import *
# from vision.unet.model import *
# from vision.utils.utils import accuracy



def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Training to detect fences')
    parser.add_argument('--exp', type=str, default='resnet_unet', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/patch_train_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='models', help='path to models directory')
    parser.add_argument('--bs', type=int, default=8, help='batch size (default: 1)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers (default: 8')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--resume_model', type=str, default='', help='path to resume model')
    parser.add_argument('--decay_margin', type=float, default=0.01, help='margin for starting decay (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.01, help='learning decay of RMSprop (default: 0.01)')
    parser.add_argument('--w_decay', type=float, default=1e-4, help='weight decay RMSprop (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of RMSprop (default: 0.9)')
    args = parser.parse_args()
    return args


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = TexelDataset(args.data_dir, 'train', True)
    val_dataset = TexelDataset(args.data_dir, 'val', False)

    model = ResNetUNet(1).to(device)

    data = faiv.DataBunch.create(dataset, val_dataset, bs=8, worker_init_fn=lambda *_: np.random.seed())

    learner = faiv.learner.Learner(data, model, loss_func=torch.nn.MSELoss())

  #  learner.load('model')
    learner.fit_one_cycle(args.epochs, args.lr, wd=args.w_decay)
    learner.save('model')


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
