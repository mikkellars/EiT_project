"""
Train UNet.
"""


import os
import sys
sys.path.append(os.getcwd())

import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from vision.unet.dataset import SynthDataset, FenceDataset
from vision.unet.model import UNet
from vision.utils.loss import FocalLoss
from vision.utils.trainer import train_model


imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Training to detect fences')
    parser.add_argument('--exp', type=str, default='synth_fence_unet', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='vision/create_fence/data/fence', help='path to data directory')
    parser.add_argument('--models_dir', type=str, default='vision/unet/models', help='path to models directory')
    parser.add_argument('--logs_dir', type=str, default='vision/unet/logs', help='path to logs directory')
    parser.add_argument('--image_dir', type=str, default='vision/unet/images')
    parser.add_argument('--bs', type=int, default=6, help='batch size (default: 1)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers (default: 8')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.001)')
    parser.add_argument('--resume_model', type=str, default='vision/unet/models/synth_fence_unet_wts.pt', help='path to resume model')
    parser.add_argument('--decay_margin', type=float, default=0.01, help='margin for starting decay')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning decay')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lr_step_size', type=int, default=3, help='Learning rate scheduler step size')
    parser.add_argument('--train', action='store_false', help='train or hyperparameters optimization')
    parser.add_argument('--hparams', type=str, default='', help='')
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.hparams != '':
        hparams = torch.load(args.hparams)
        args.lr = hparams['params']['lr']
        args.wd = hparams['params']['wd']
        args.bs = int(hparams['params']['bs'])
        print(f'Learning rate: {args.lr:.4f}. Weight decay: {args.wd:.4f}. Batch size: {args.bs}')

    # --------
    # Training
    # --------

    fence_path = 'vision/data/fence_data/train_set'
    synth_path = 'vision/create_fence/data/fence'

    train_data_1 = FenceDataset(fence_path, 'train', get_transform(train=True))
    train_data_2 = SynthDataset(synth_path, 'train', get_transform(train=True))
    train_data = torch.utils.data.ConcatDataset([train_data_1, train_data_2])
    train_dl = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    
    val_data_1 = FenceDataset(fence_path, 'val', get_transform(train=False))
    val_data_2 = SynthDataset(synth_path, 'val', get_transform(train=False))
    val_data = torch.utils.data.ConcatDataset([val_data_1, val_data_2])
    val_dl = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    
    dataloaders = {'train': train_dl, 'val': val_dl}

    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=3,
        out_channels=2,
        init_features=32,
        pretrained=False
    ).to(device)
    
    if args.resume_model != '':
        wts = torch.load(args.resume_model)
        model.load_state_dict(wts)
        print(f'Loaded model from {args.resume_model}')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, verbose=True)

    model, acc, loss = train_model(model, criterion, dataloaders, optimizer, n_classes=1,
                                   name=args.exp, log_path=args.logs_dir, epochs=args.epochs,
                                   scheduler=scheduler)

    torch.save(model.state_dict(), f'{args.models_dir}/{args.exp}_wts.pt')

    # -------
    # Testing
    # -------

    test_data = FenceDataset('vision/data/fence_data/test_set', 'test', get_transform(train=False))
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers)
    loop = tqdm(test_dl, leave=False)
    losses, accs = list(), list()
    for i, data in enumerate(loop):
        img, mask = data
        model.eval()
        with torch.no_grad():
            pred = model(img.to(device))
        loss = criterion(pred, mask.to(device))
        losses.append(loss.item())
        pred = pred.argmax(dim=1).float()
        acc = (pred == mask.to(device)).float().mean().item()
        accs.append(acc)
        save_image_w_mask(img[0], pred[0])
        plt.savefig(f'{args.image_dir}/{i:03d}.png')
        plt.close('all')
        loop.set_description(f'[Loss {np.mean(losses):.4f}(+/-{np.std(losses):.4f})] [Accuracy {np.mean(accs):.4f}(+/-{np.std(accs):.4f})]')

    print(f'Testing done! Loss {np.mean(losses):.4f}(+/-{np.std(losses):.4f}) and accuracy {np.mean(accs):.4f}(+/-{np.std(accs):.4f})')


def get_h_params(args):
    from bayes_opt import BayesianOptimization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = FenceDataset(args.data_dir, 'train', True)
    val_data = FenceDataset(args.data_dir, 'val', False)

    def fit_with(lr:float, wd:float, bs:float):
        train_dl = DataLoader(train_data, batch_size=int(bs), shuffle=True, num_workers=args.workers)
        val_dl = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers)
        dataloaders = {'train': train_dl, 'val': val_dl}

        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=3,
            out_channels=2,
            init_features=32,
            pretrained=False
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss().to(device)

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


def get_transform(train:bool, im_size:int=400):
    if train:
        aug = A.Compose([
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(224, 720), height=im_size, width=im_size, p=0.5),
                A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_CUBIC, p=0.5)
            ], p=1), 
            A.ChannelShuffle(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Blur(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        aug = A.Compose([
            A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ])
    return aug


def plot_image_w_mask(img, pred)->None:
    fig, ax = plt.subplots(1)
    img = T.Normalize((-imagenet_mean/imagenet_std).tolist(), (1.0/imagenet_std).tolist())(img)
    img = T.ToPILImage()(img)
    ax.imshow(img)
    pred = pred.mul(255).cpu()
    ax.imshow(T.ToPILImage()(pred), alpha=0.5)
    plt.axis('off')
    plt.show()


def save_image_w_mask(img, pred)->None:
    fig, ax = plt.subplots(1)
    img = T.Normalize((-imagenet_mean/imagenet_std).tolist(), (1.0/imagenet_std).tolist())(img)
    img = T.ToPILImage()(img)
    ax.imshow(img)
    pred = pred.mul(255).cpu()
    ax.imshow(T.ToPILImage()(pred), alpha=0.5)
    plt.axis('off')


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    if args.train: train(args)
    else: get_h_params(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s.')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = torch.hub.load(
    #     'mateuszbuda/brain-segmentation-pytorch',
    #     'unet',
    #     in_channels=3,
    #     out_channels=2,
    #     init_features=32,
    #     pretrained=False
    # ).to(device)
    
    # wts = torch.load('vision/unet/models/synth_fence_unet_wts.pt')
    # model.load_state_dict(wts)

    # transforms = get_transform(train=False)

    # im_path = '/home/mathias/Desktop'
    # images = [f'{im_path}/{f}' for f in os.listdir(im_path)]

    # for path in images:
    #     im = cv2.imread(path, cv2.IMREAD_COLOR)
    #     im = transforms(image=im)['image']
    #     im = im.unsqueeze(0)

    #     pred = model(im.to(device))
    #     pred = pred.argmax(dim=1).float()
    #     plot_image_w_mask(im[0], pred[0])
    #     plt.close('all')