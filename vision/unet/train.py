"""
Train UNet.
"""


import os
import sys
sys.path.append(os.getcwd())

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from vision.unet.dataset import TexelDataset, FenceDataset
from vision.unet.model import UNet
from vision.utils.loss import FocalLoss
from vision.utils.trainer import train_model


imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Training to detect fences')
    parser.add_argument('--train', action='store_false')
    parser.add_argument('--exp', type=str, default='unet_25', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/train_set', help='path to data directory')
    parser.add_argument('--models_dir', type=str, default='vision/unet/models', help='path to models directory')
    parser.add_argument('--logs_dir', type=str, default='vision/unet/logs', help='path to logs dir')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--resume_model', type=str, default='vision/unet/models/unet_25_wts.pt', help='path to resume model')
    parser.add_argument('--decay_margin', type=float, default=0.01, help='margin for starting decay')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning decay')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lr_step_size', type=int, default=3, help='Learning rate scheduler step size')
    args = parser.parse_args()
    return args


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------
    # Training
    # --------

    train_data = FenceDataset(args.data_dir, 'train', True)
    train_dl = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    val_data = FenceDataset(args.data_dir, 'val', False)
    val_dl = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    dataloaders = {'train': train_dl, 'val': val_dl}

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=2, init_features=32, pretrained=False).to(device)
    
    if args.resume_model != '':
        wts = torch.load(args.resume_model)
        model.load_state_dict(wts)

    criterion = FocalLoss(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

    model, acc, loss = train_model(model, criterion, dataloaders, optimizer, n_classes=1,
                                   name=args.exp, log_path=args.logs_dir, epochs=args.epochs,
                                   scheduler=scheduler)

    torch.save(model.state_dict(), f'{args.models_dir}/{args.exp}_wts.pt')

    # -------
    # Testing
    # -------

    test_data = FenceDataset('vision/data/fence_data/test_set', 'test', False)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers)
    loop = tqdm(test_dl)
    loop.set_description('Show the test results..')
    for data in loop:
        img, _ = data
        model.eval()
        with torch.no_grad():
            pred = model(img.to(device))
        pred = pred.argmax(dim=1).float()
        plot_image_w_mask(img[0], pred[0])


def get_h_params(args):
    from bayes_opt import BayesianOptimization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = FenceDataset(args.data_dir, 'train', True)
    val_data = FenceDataset(args.data_dir, 'val', False)

    def fit_with(lr:float, wd:float, bs:float):
        train_dl = DataLoader(train_data, batch_size=int(bs), shuffle=True, num_workers=args.workers)
        val_dl = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers)
        dataloaders = {'train': train_dl, 'val': val_dl}
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=3, out_channels=1, init_features=32,
                                pretrained=True).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model, acc, loss = train_model(
            model=model,
            criterion=criterion,
            dls=dataloaders,
            opt=optimizer, 
            n_classes=2,
            name=f'h_param_{lr:.5f}_{wd:.5f}',
            log_path=args.logs_dir,
            epochs=args.epochs,
            verbose=False
        )
        return acc

    pbounds = {'lr': (1e-4, 1e-2), 'wd': (1e-4, 0.4), 'bs': (1, 4)}
    optimizer = BayesianOptimization(
        f=fit_with,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(n_iter=10)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)
    filename = f'{args.models_dir}/hparams.pt'
    print(f'Saving hyperparameters at {filename}')
    torch.save(optimizer.max, filename)


def plot_image_w_mask(img, pred)->None:
    fig, ax = plt.subplots(1)
    img = transforms.Normalize((-imagenet_mean/imagenet_std).tolist(), (1.0/imagenet_std).tolist())(img)
    img = transforms.ToPILImage()(img)
    ax.imshow(img)
    pred = pred.mul(255).cpu()
    ax.imshow(transforms.ToPILImage()(pred), alpha=0.5)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    if args.train: train(args)
    else: get_h_params(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s.')
