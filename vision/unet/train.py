"""
"""


import os
import time
import datetime
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from model import *

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Training to detect fences')
    parser.add_argument('--exp', type=str, default='detectnet', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/train_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='vision/unet/models', help='path to models directory')
    parser.add_argument('--bs', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers (default: 8')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--resume_model', type=str, default='', help='path to resume model')
    parser.add_argument('--decay_margin', type=float, default=0.01, help='margin for starting decay (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.01, help='learning decay of RMSprop (default: 0.01)')
    parser.add_argument('--w_decay', type=float, default=1e-8, help='weight decay RMSprop (default: 1e-8)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of RMSprop (default: 0.9)')
    args = parser.parse_args()
    return args


def main(args):
    best_val_loss = float('inf')
    args.start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Datasets
    dataset = DetectNetDataset(args.data_dir, 'train', True)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers)

    val_dataset = DetectNetDataset(args.data_dir, 'val', False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    dataloaders = {'train': dataloader, 'val': val_dataloader}

    # Model
    model = UNet(15, 1).to(device)

    if args.resume_model != '':
        saved_model = torch.load(args.resume_model)
        model.load_state_dict(saved_model["model_state_dict"])
        best_val_loss = float(saved_model["loss"])
        print(f'Loaded {args.resume_model}. The model is trained for {saved_model["epoch"]} epochs with {saved_model["loss"]} loss')

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.w_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)

    # Loss
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    loop = tqdm(range(args.start_epoch, args.epochs))
    start_time = time.time()
    for epoch in loop:

        for phase in ['train', 'val']:

            phase_loss = []
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for i, data in enumerate(dataloaders[phase]):
                # img, sobel_x, sobel_y, laplace_4, laplace_8, mask = data
                # img = img.to(device, dtype=torch.float32)
                # sobel_x = sobel_x.to(device, dtype=torch.float32)
                # sobel_y = sobel_y.to(device, dtype=torch.float32)
                # laplace_4 = laplace_4.to(device, dtype=torch.float32)
                # laplace_8 = laplace_8.to(device, dtype=torch.float32)

                img, mask = data

                img = img.to(device, dtype=torch.float32)
                
                mask_type = torch.float32 if model.n_classes == 1 else torch.long
                mask = mask.to(device, dtype=mask_type)
                mask = mask.view(mask.size(0), 1, mask.size(1), mask.size(2))

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(img)
                    loss = criterion(pred, mask)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                phase_loss.append(loss.item())

                loop.set_description(f'[Epoch {epoch+1:03d}/{args.epochs:03d}] [Phase {phase}] [Batch {i+1:03d}/{len(dataloaders[phase]):03d}] [Loss {np.mean(phase_loss):.8f}]')

            if phase == 'val':
                scheduler.step(np.mean(phase_loss))

                if np.mean(phase_loss) < best_val_loss:
                    best_val_loss = np.mean(phase_loss)
                    loop.write(f'New model found at epoch {epoch+1} with loss {best_val_loss:.8f}')

                    # Save model
                    model_path = f'{args.save_dir}/{args.exp}_model.pt'
                    torch.save({'epoch': epoch+1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': best_val_loss},
                                model_path)

        # Save checkpoint after each epoch
        model_path = f'{args.save_dir}/{args.exp}_checkpoint.pt'
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.mean(phase_loss)},
                    model_path)
    
    end_time = datetime.timedelta(seconds=np.ceil(time.time() - start_time))    
    print(f'Training took {end_time}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
