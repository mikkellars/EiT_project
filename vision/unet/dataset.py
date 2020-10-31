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
        self.resize = transforms.Resize((256, 256))
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
        img = self.resize(img)
        if self.transform:
            img = self.clr_jitter(img)
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('L')
        mask = self.resize(mask)
        img, mask = np.array(img), np.array(mask)
        mask = np.where(mask!=0, 1, 0)
        if self.transform:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() < 0.5:
                k = random.randint(1, 3) * 2 + 1
                img = cv2.blur(img, (k, k))
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
        self.resize = transforms.Resize((400, 400))
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
        # mask = torch.nn.functional.one_hot(mask, 2).squeeze(0)
        # mask = mask.permute(2, 0, 1)
        return img, mask


def one_hot(mask:torch.Tensor)->torch.Tensor:
    one_hot_map = list()
    for c in range(n_classes):
        c_map = torch.equal(mask, c)
        one_hot_map.append(c_map)
    one_hot_map = torch.stack(one_hot_map, dim=-1).float()


if __name__ == '__main__':
    print(__doc__)

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    def show(imgs):
        plt.figure(figsize=(6,3))
        for i, img in enumerate(imgs):
            plt.subplot(len(imgs) // 2, 2, i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    data_dir = 'vision/data/fence_data/patch_train_set'
    dataset = TexelDataset(data_dir, 'train', transforms=True)

    loop = tqdm(dataset)
    for i, data in enumerate(loop):

        img, mask = data
        # img, sobel_x, sobel_y, laplace_4, laplace_8, mask = data

        img = img.data.cpu().detach().numpy().astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))

        mask = mask.data.cpu().detach().numpy().astype(np.uint8)

        # sobel_x = sobel_x.data.cpu().detach().numpy().astype(np.uint8)
        # sobel_x = np.transpose(sobel_x, (1, 2, 0))

        # sobel_y = sobel_y.data.cpu().detach().numpy().astype(np.uint8)
        # sobel_y = np.transpose(sobel_y, (1, 2, 0))


        # laplace_4 = laplace_4.data.cpu().detach().numpy().astype(np.uint8)
        # laplace_4 = np.transpose(laplace_4, (1, 2, 0))

        # laplace_8 = laplace_8.data.cpu().detach().numpy().astype(np.uint8)
        # laplace_8 = np.transpose(laplace_8, (1, 2, 0))

        # show([img, mask, sobel_x, sobel_y, laplace_4, laplace_8])
        show([img, mask])
    
    print('Done with test!')