"""
Module for creating a Dataset class to fetch the fence data.
"""


import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


classes = (
    'background',
    'fence'
)

n_classes = len(classes)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


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
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, index:int):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     img = self.clr_jitter(img)
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('L')
        # mask = self.to_grayscale(mask)
        img, mask = np.array(img), np.array(mask)
        if self.transform:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            # if random.random() < 0.5:
            #     k = random.randint(1, 3) * 2 + 1
            #     img = cv2.blur(img, (k, k))
        img, mask = self.to_tensor(img), self.to_tensor(mask)
        img = self.norm(img)
        return img, mask


class FenceDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, mode:str='train', transform:bool=True):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.clr_jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
        self.to_grayscale = transforms.Grayscale(1)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(*imagenet_stats)
        self.resize = transforms.Resize((400,400))
        if mode == 'train':
            files = open(f'{root}/train.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}.jpg' for f in files]
            self.masks = [f'{root}/labels/{f}.png' for f in files]
        elif mode == 'val':
            files = open(f'{root}/val.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}.jpg' for f in files]
            self.masks = [f'{root}/labels/{f}.png' for f in files]
        elif mode == 'test':
            img_files = os.path.join(root, 'images')
            imgs = [os.path.join(img_files, f) for f in os.listdir(img_files)]
            self.imgs = sorted(imgs)
            mask_files = os.path.join(root, 'labels')
            masks = [os.path.join(mask_files, f) for f in os.listdir(mask_files)]
            self.masks = sorted(masks)
        else: raise NotImplementedError()
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, index:int):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img = self.resize(img)
        if self.transform:
            img = self.clr_jitter(img)
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('L')
        mask = self.resize(mask)
        img, mask = np.array(img), np.array(mask)
        if self.transform:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() < 0.5:
                k = random.randint(1, 3) * 2 + 1
                img = cv2.blur(img, (k, k))
        mask = np.where(mask!=0, 1, 0)
        img, mask = self.to_tensor(img), self.to_tensor(mask).long().squeeze(0)
        img = self.norm(img)
        return img, mask


if __name__ == '__main__':
    print(__doc__)

    import time
    from tqdm import tqdm

    data_dir = '../data/fence_data/patch_train_set'
    dataset = TexelDataset(data_dir, 'train', transforms=True)

    times = list()

    loop = tqdm(range(30))
    start_time = time.time()
    for i in loop:
        data = random.choice(dataset)
        img, mask = data
        times.append((time.time() - start_time))
        start = time.time()
    
    print(f'Done with test! Collecting data took in average {np.mean(times):.4f} seconds.')