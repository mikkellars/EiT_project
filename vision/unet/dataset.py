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
    def __init__(self, root:str, mode:str='train', transforms=None):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        if mode == 'train':
            files = open(f'{root}/train.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}' for f in files]
            self.masks = [f'{root}/labels/{f}' for f in files]
            self.masks = [f.replace('.jpg', '.png') for f in self.masks]
        elif mode == 'val':
            files = open(f'{root}/val.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}' for f in files]
            self.masks = [f'{root}/labels/{f}' for f in files]
            self.masks = [f.replace('.jpg', '.png') for f in self.masks]
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
        im_file = self.imgs[index]
        im = cv2.imread(im_file, cv2.IMREAD_COLOR)
        assert im is not None, f'Image is none, smothing went wrong with data loading.'

        mask_file = self.masks[index]
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'Mask is none, smothing went wrong with data loading.'
        mask = np.where(mask != 0, 1, 0)

        if self.transforms is not None:
            aug = self.transforms(image=im, mask=mask)
            im = aug['image']
            mask = aug['mask']

        return im, mask.long()


class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, mode:str='train', transforms=None):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        if mode == 'train':
            files = open(f'{root}/train.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/cycles/{f}.png' for f in files]
            self.masks = [f'{root}/depth/{f}.png' for f in files]
        elif mode == 'val':
            files = open(f'{root}/val.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/cycles/{f}.png' for f in files]
            self.masks = [f'{root}/depth/{f}.png' for f in files]
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
        im_file = self.imgs[index]
        im = cv2.imread(im_file, cv2.IMREAD_COLOR)
        assert im is not None, f'Image is none, smothing went wrong with data loading.'

        mask_file = self.masks[index]
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'Mask is none, smothing went wrong with data loading.'
        mask = np.where(mask != 0, 1, 0)

        if self.transforms is not None:
            aug = self.transforms(image=im, mask=mask)
            im = aug['image']
            mask = aug['mask']

        return im, mask.long()


def one_hot(mask:torch.Tensor)->torch.Tensor:
    one_hot_map = list()
    for c in range(n_classes):
        c_map = torch.equal(mask, c)
        one_hot_map.append(c_map)
    one_hot_map = torch.stack(one_hot_map, dim=-1).float()


if __name__ == '__main__':
    print(__doc__)

    import albumentations as A
    import matplotlib.pyplot as plt
    from albumentations.pytorch import ToTensorV2

    def get_transform(train:bool, im_size:int=400):
        if train:
            aug = A.Compose(
                [
                    A.OneOf([
                        A.RandomSizedCrop(min_max_height=(224, 720), height=im_size, width=im_size, p=0.5),
                        A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_CUBIC, p=0.5)
                    ], p=1), 
                    A.ChannelShuffle(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.5),
                    # A.OneOf([
                    #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    #     A.GridDistortion(p=0.5),
                    #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
                    # ], p=0.8),
                    A.Blur(p=0.5),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            aug = A.Compose(
                [
                    A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_CUBIC),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        return aug

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.close('all')

    def visualize(image, mask):
        image = np.transpose(image.mul(255).numpy(), (1,2,0))
        mask = mask.mul(255).numpy()
        fontsize = 18
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap='gray')
        plt.show()
        plt.close('all')

    data_dir = 'vision/data/fence_data/train_set'
    dataset = FenceDataset(data_dir, 'train', transforms=get_transform(train=False))

    for i, data in enumerate(dataset):
        if i == 32: break
        img, mask = data
        visualize(img, mask)
    
    print('Done with test!')