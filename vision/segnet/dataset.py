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


class FenceDataset(Dataset):
    """Fence Dataset.

    Args:
        root (str): Path to data.
        mode (str, optional): Mode. Defaults to 'train'.
        transforms (bool, optional): Performs transformations. Defaults to False.

    Raises:
        NotImplementedError: if mode unknown.
    """
    def __init__(self, root: str, mode: str = 'train', transforms: bool = False,
                 use_synth: bool = False):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.use_synth = use_synth
        self.resize = T.Resize(size=(512, 512))

        if mode == 'train':
            files = open(f'{root}/train.txt', 'rt').read().split('\n')[:-1]
            self.imgs = [f'{root}/images/{f}.jpg' for f in files]
            self.masks = [f'{root}/labels/{f}.png' for f in files]
            if use_synth:
                synth_files = files = open('synth_ml_data/train.txt', 'rt').read().split('\n')[:-1]
                synth_imgs = [f'synth_ml_data/cycles/{f}.png' for f in synth_files]
                synth_masks = [f'synth_ml_data/object_index/{f}.png' for f in synth_files]
                self.imgs.extend(synth_imgs)
                self.masks.extend(synth_masks)
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
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        # Apply color transformation
        if self.transforms is True and random.random() > 0.5:
            trancolor = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5)
            img = trancolor(img)

        img = np.array(img)

        # Load mask
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)

        # Data augmentation
        if self.transforms is True:

            # Random horizontal flipping
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

            # # Random crop
            # if random.random() > 0.0:
            #     img_size = random.randint(512, img.shape[0])
            #     img = Image.fromarray(img)
            #     mask = Image.fromarray(mask)
            #     i, j, h, w = T.RandomCrop.get_params(img, output_size=(img_size, img_size))
            #     img = F.crop(img, i, j, h, w)
            #     mask = F.crop(mask, i, j, h, w)
            #     img = np.array(img)
            #     mask = np.array(mask)

        # Convert mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask != 0] = 1

        # Resize
        img = np.array(self.resize(Image.fromarray(img)))
        mask = np.array(self.resize(Image.fromarray(mask)))

        # To tensor
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))

        return img, mask


if __name__ == '__main__':
    print(__doc__)

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    def show(img, mask):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')
        plt.tight_layout(True)
        plt.show()

    data_dir = 'data/fence_data/train_set'
    dataset = FenceDataset(data_dir, 'train', transforms=True)

    loop = tqdm(dataset)
    for i, data in enumerate(loop):
        img, mask = data
        img = img.data.cpu().detach().numpy().astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        mask = mask.data.cpu().detach().numpy().astype(np.uint8)
        show(img, mask)
