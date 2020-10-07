import os
import time
import datetime
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from model import *

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Training to detect fences')
    parser.add_argument('--exp', type=str, default='segnet', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='data/fence_data/train_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='models', help='path to models directory')
    parser.add_argument('--bs', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers (default: 8')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--resume_model', type=str, default='models/model.pt', help='path to resume model')
    parser.add_argument('--decay_margin', type=float, default=0.01, help='margin for starting decay (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.01, help='learning decay (default: 0.01)')
    args = parser.parse_args()
    return args


def main(args):
    args.decay_start = False
    args.start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Datasets
    dataset = FenceDataset(args.data_dir, 'train', True)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers)

    val_dataset = FenceDataset(args.data_dir, 'val', False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    dataloaders = {'train': dataloader, 'val': val_dataloader}

    # Model
    model = Segnet(3, n_classes).to(device)

    if args.resume_model != '':
        saved_model = torch.load(args.resume_model)
        model.load_state_dict(saved_model["model_state_dict"])
        print(f'Loaded {args.resume_model}. The model is trained for {saved_model["epoch"]} epochs with {saved_model["loss"]} loss')

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Loss
    criterion = Loss()

    loop = tqdm(range(args.start_epoch, args.epochs))
    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in loop:

        for phase in ['train', 'val']:

            phase_loss = 0.0
            phase_time = 0
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for img, mask in dataloaders[phase]:
                phase_time += 1

                img = Variable(img).to(device)
                mask = Variable(mask).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(img)
                    loss = criterion(pred, mask)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                phase_loss += loss.item()

                loop.set_description(f'[Epoch {epoch+1:03d}/{args.epochs:03d}] [Phase {phase}] [Batch {phase_time:03d}/{len(dataloaders[phase]):03d}] [Loss {loss.item():.8f}]')

            phase_loss = phase_loss / phase_time

            if phase == 'val' and phase_loss < best_val_loss:
                best_val_loss = phase_loss
                loop.write(f'New model found at epoch {epoch+1} with loss {best_val_loss:.8f}')

                # Save model
                model_path = f'{args.save_dir}/{args.exp}_model.pt'
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss
                }, model_path)

        # Learning decay
        if args.decay_start is False and best_val_loss < args.decay_margin:
            args.decay_start = True
            args.lr *= args.lr_decay
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=args.lr)
    
    end_time = datetime.timedelta(seconds=np.ceil(time.time() - start_time))    
    print(f'Training took {end_time}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
