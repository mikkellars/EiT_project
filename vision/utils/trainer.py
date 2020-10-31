"""
"""

import os
import sys
sys.path.append(os.getcwd())

import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from vision.utils.metrics import AverageMeter, accuracy, intersection_over_union


def train_model(model, criterion, dls, opt, n_classes:int,
                name:str, log_path:str, epochs:int=50, verbose:bool=True,
                scheduler=None):
    """Train a segmentation model.

    Args:
        model (torch.nn.Module): Deep neural network model to train.
        criterion (torch.nn.Loss): Loss func to use.
        dls (dict): Dictionary of dataloaders.
        opt (torch.optim): Optimizer function to use.
        bpath (str): [description]
        epochs (int, optional): Numper of epochs to run. Defaults to 50.
        scheduler (torch.optim.Scheduler, optional): Learning rate scheduler

    Returns:
        torch.nn.Module, float, float: The train model with the best weights,
            the best obtained acc, the best obtained loss.
    """

    writer_train = SummaryWriter(f'{log_path}/train-{name}')
    writer_val = SummaryWriter(f'{log_path}/val-{name}')
    writers = {"train": writer_train, "val": writer_val}

    start_time = time.time()

    best_loss = 1e10
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    loop = tqdm(range(1, epochs+1)) if verbose else range(1, epochs+1)
    for epoch in loop:
        if verbose:
            loop.write('-' * 50)
            loop.write(f'[Epoch {epoch}/{epochs}]')

        for phase in ['train', 'val']:
            loss_meter.reset()
            acc_meter.reset()

            if phase == 'train': model.train()
            else: model.eval()

            for inputs, masks in iter(dls[phase]):
                inputs = inputs.to(device)
                masks = masks.to(device)

                opt.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    outputs = outputs['out']
                    loss = criterion(outputs, masks)

                    if phase == 'train':
                        loss.backward()
                        opt.step()

                bs = inputs.size(0)
                loss_meter.update(loss.item(), bs)
                # outputs = torch.sigmoid(outputs)
                # outputs = (outputs > 0.5).float()
                # acc = (torch.sum(outputs == masks).float() / masks.nelement())
                acc = (outputs.argmax(dim=1) == masks).float().mean()
                acc_meter.update(acc.item(), bs)
                    
            if verbose: loop.write(f'[Phase {phase}]\t[Loss {loss_meter.avg:.4f}] [Accuracy {acc_meter.avg:.3f}]')

            tensorboard_log_scalar(writers[phase], name, epoch, loss_meter.avg, acc_meter.avg)

            if phase == 'val' and scheduler is not None:
                scheduler.step(acc_meter.avg)

            # if phase == 'val' and loss_meter.avg <= best_loss and acc_meter.avg >= best_acc:
            if phase == 'val' and acc_meter.avg > best_acc:
                best_loss = loss_meter.avg
                best_acc = acc_meter.avg
                best_model_wts = copy.deepcopy(model.state_dict())
                if verbose: loop.write(f'New model found at epoch {epoch} with loss {best_loss:.6f} and accuracy {best_acc:.6f}!')

    end_time = time.time() - start_time
    if verbose: print(f'Training complete in {end_time//60:.0f}m {end_time%60:.0f}s with best loss {best_loss:.6f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc, best_loss


def tensorboard_log_scalar(writer, title:str, epoch, loss, acc=None, iou=None):
    """Function to log scalar to tensorboard as graph

    Args:
        writer ([SummaryWriter]): [tensorboard writer]
        overall_title ([string]): [overall title / group title shown on tensorboard]
        epoch ([int]): [current training epoch]
        epoch_loss ([float]): [current epoch loss]
        epoch_acc ([float]): [current accuracy loss]
    """
    writer.add_scalar(f'{title}/Loss', loss, epoch)
    if acc is not None: writer.add_scalar(f'{title}/Accuracy', acc, epoch)
    if iou is not None: writer.add_scalar(f'{title}/IoU', iou, epoch)