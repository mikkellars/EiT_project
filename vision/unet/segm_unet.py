

import os
import sys
import cv2
import copy
import time
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
from torchsummary import summary
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import make_grid
from lib.unet import UNet
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('background', 'fence')
n_classes = len(classes)
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
inv_imagenet_stats = ([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])


def show(img:torch.Tensor)->None:
    img = img.mul(255.0)
    img = img.data.cpu().detach().numpy().astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    plt.axis('off')
    plt.imshow(img, interpolation='nearest')
    plt.show()


class TexelDataset(Dataset):
    
    def __init__(self, root:str, mode:str='train', transform:bool=True):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.clr_jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
        self.to_grayscale = transforms.Grayscale(1)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(*imagenet_stats)
        if mode == 'train':
            files = open(f'{root}/train.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}.png' for f in files]
            self.masks = [f'{root}/labels/{f}.png' for f in files]
        elif mode == 'val':
            files = open(f'{root}/val.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}.png' for f in files]
            self.masks = [f'{root}/labels/{f}.png' for f in files]
        elif mode == 'test':
            img_files = os.path.join(root, 'images')
            imgs = [os.path.join(img_files, f) for f in os.listdir(img_files)]
            self.imgs = sorted(imgs)
            mask_files = os.path.join(root, 'labels')
            masks = [os.path.join(mask_files, f) for f in os.listdir(mask_files)]
            self.masks = sorted(masks)
        else: raise NotImplementedError()
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index:int):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.clr_jitter(img)
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = self.to_grayscale(mask)
        img, mask = np.array(img), np.array(mask)
        if self.transform:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() < 0.5:
                k = random.randint(1, 3) * 2 + 1
                img = cv2.blur(img, (k, k))
        sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        img = self.to_tensor(img)
        sobel_x = self.to_tensor(sobel_x)
        sobel_y = self.to_tensor(sobel_y)
        mask = self.to_tensor(mask)
        # img = self.norm(img)
        # return img, mask
        return torch.cat([img, sobel_x, sobel_y], dim=0), mask


data_path = 'vision/data/fence_data/patch_train_set'

train_ds = TexelDataset(data_path, 'train', True)
train_dl = DataLoader(train_ds, 4, True, num_workers=8)

val_ds = TexelDataset(data_path, 'val', False)
val_dl = DataLoader(val_ds, 1, False, num_workers=8)

dls = {'train': train_dl, 'val': val_dl}


images = list()
for i, data in enumerate(dls['val']):
    if i == 8*3: break
    x, y = data
    x, y = x.squeeze(0), y.squeeze(0)
    # x = transforms.Normalize(*inv_imagenet_stats)(x)
    x = x[:3, :, :]
    y = y.repeat(3, 1, 1)
    images.append(x)
    images.append(y)

show(make_grid(images, padding=25))


# model = ResNetUNet(1).to(device)
model = UNet(9, 1).to(device)
# summary(model, input_size=(9, 256, 256))


def dice_loss(pred, targ, smooth:float=1.0):
    pred, targ = pred.contiguous(), targ.contiguous()
    intersection = (pred * targ).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + targ.sum(dim=2).sum(dim=2)
    loss = (1.0 - ((2.0*intersection+smooth)/(union+smooth)))
    return loss.mean()

def calc_loss(pred, targ, bce_wt:float=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, targ)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, targ)
    loss = bce * bce_wt + dice * (1 - bce_wt)
    return bce, dice, loss

def fast_hist(pred, targ, n_cls:int):
    mask = (targ >= 0) & (targ < n_cls)
    hist = torch.bincount(n_cls*targ[mask]+pred[mask], minlength=n_cls**2)
    hist = hist.reshape(n_cls, n_cls).float()
    return hist

def overall_pixel_acc(hist):
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + 1e-10)
    return overall_acc

def per_cls_pixel_acc(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + 1e-10)
    # avg_per_class_acc = nanmean(per_class_acc)
    avg_per_class_acc = torch.mean(per_class_acc[per_class_acc == per_class_acc])
    return avg_per_class_acc

def accuracy(pred, targ):
    valid = (targ >= 0).float()
    acc_sum = (valid * (pred == targ)).sum()
    acc = acc_sum.float() / (valid.sum() + 1e-10)
    return acc


def train_model(model, optimizer, criterion, scheduler, epochs:int=30):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # loop = tqdm(range(1, epochs+1))
    for epoch in range(1, epochs+1):
        print('-'*50)
        print(f'Epoch {epoch}/{epochs}')
        
        epoch_start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            
            metrics = {'bce': 0.0, 'dice': 0.0, 'loss': 0.0, 'acc': 0.0}
            samples = 0
            
            for inputs, labels in dls[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                
                samples += inputs.size(0)
                metrics['loss'] += loss.item()
                outputs = (outputs > 0.5).float()
                metrics['acc'] += (torch.sum(outputs == labels).float() / labels.nelement())

            loss = metrics['loss'] / samples
            acc = metrics['acc'] / samples

            print(f'[Phase {phase}]\t[Loss {loss:.4f}] [Acc {acc:.4f}]')
            
            if phase == 'val' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New model found at epoch {epoch} with a loss of {loss:.4f}')
            
        epoch_end = time.time() - epoch_start
        print(f'Epoch {epoch} took {epoch_end//60:.0f} m {epoch_end%60:.1f} s.')
    
    print(f'Best val loss {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# Freeze backbone layers
#for l in model.base_layers:
#    for param in l.parameters():
#        param.requires_grad = False

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
model = train_model(model, optimizer, criterion, exp_lr_scheduler, 30)
torch.save(model.state_dict(), 'vision/models/segm_unet_30.pt')


test_ds = TexelDataset(data_path, 'test', False)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8)
criterion = torch.nn.BCEWithLogitsLoss()
accs, losses = list(), list()
loop = tqdm(test_dl)
start_time = time.time()
for i, data in enumerate(loop):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    losses.append(loss.item())
    outputs = (outputs > 0.5).float()
    acc = (torch.sum(outputs == labels).float() / labels.nelement())
    accs.append(acc.item())
    loop.set_description(f'[Batch {i+1:03d}/{len(test_dl)}] [Loss {loss.item():.3f}] [Acc {acc:.3f}]')
end_time = time.time() - start_time
m_acc = np.mean(accs)
m_losses = np.mean(losses)
print(f'Accuracy {np.mean(accs):.3f} +/-{np.std(accs):.3f}')
print(f'Loss: {np.mean(losses):.3f} +/-{np.std(losses):.3f}')
print(f'Done with testing! It took {end_time//60} m {end_time%60:.1f} s')


images = list()
for i, data in enumerate(test_dl):
    if i == 8*5: break
    x, y_true = data
    x, y_true = x.to(device), y_true.to(device)
    with torch.no_grad():
        y_pred = model(x)
    x = x.squeeze(0)[:3, :, :]
    y_pred = (y_pred > 0.5).float().squeeze(0).repeat(3, 1, 1)
    # y_pred = (y_pred > 0).float().squeeze(0).repeat(3, 1, 1)
    images.append(x)
    images.append(y_pred)

show(make_grid(images, padding=25))

inp_path = 'vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_16.jpg'
inp = Image.open(inp_path).convert('RGB')
inp = np.array(inp)
plt.axis('off')
plt.imshow(inp)


def show_patches(images:list)->None:
    fig = plt.figure(figsize=(8,8))
    for i in range(cols*rows):
        img = images[i]
        fig.add_subplot(rows, cols, i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()


rois = list()
height, width, channels = inp.shape
rows, cols = height // 128, width // 128
for i in range(rows):
    for j in range(cols):
        ymin = i * height // rows
        ymax = i * height // rows + height // rows
        xmin = j * width // cols
        xmax = j * width // cols + width // cols
        roi = inp[ymin: ymax, xmin: xmax]
        roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_CUBIC)
        rois.append(roi)

show_patches(rois)

preds = list()
for x in rois:
    model.eval()
    sobel_x = cv2.Sobel(x, cv2.CV_16S, 1, 0)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(x, cv2.CV_16S, 0, 1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_x = transforms.ToTensor()(sobel_x)
    sobel_y = transforms.ToTensor()(sobel_y)
    x = transforms.ToTensor()(x)
    #x = transforms.Normalize(*imagenet_stats)(x)
    x = torch.cat([x, sobel_x, sobel_y], dim=0)
    x = x.to(device).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    y = (y > 0.5).float().squeeze(0).repeat(3, 1, 1).mul(255.0)
    y = y.data.cpu().detach().numpy()
    y = y.astype(np.uint8)
    y = np.transpose(y, (1,2,0))
    preds.append(y)
    
show_patches(preds)