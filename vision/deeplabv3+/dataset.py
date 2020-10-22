"""
Module for creating a Dataset class to fetch the fence data.
"""


import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


classes = (
    'background',
    'fence'
)

n_classes = len(classes)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class TexelDataset(Dataset):

    def __init__(self, root: str, mode: str = 'train', transforms: bool = False):

        self.root = root
        self.mode = mode
        self.transforms = transforms

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
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        # Apply color transformation
        if self.transforms and random.random() < 0.5:
            img = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)(img)

        img = np.array(img)

        # Load mask
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)

        # Data augmentation
        if self.transforms:

            # Random horizontal flipping
            if random.random() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # Random vertical flipping
            if random.random() < 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

            if random.random() < 0.5:
                k = random.randint(1, 3) * 2 + 1
                img = cv2.blur(img, (k, k))

        # Convert mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask != 0] = 1

        # To tensor
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))

        img = T.Normalize(*imagenet_stats)(img)
        
        # mask_type = torch.float32 if n_classes == 1 else torch.long
        mask = mask.to(torch.float32).unsqueeze(0)

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