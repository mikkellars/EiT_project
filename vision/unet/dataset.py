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
    def __init__(self, root: str, mode: str = 'train', transforms: bool = False):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.resize = T.Resize(size=(512, 512))

        if mode == 'train':
            # TODO: add synthetic data
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
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        # Apply color transformation
        if self.transforms and random.random() > 0.5:
            trancolor = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5)
            img = trancolor(img)

        img = np.array(img)

        # Load mask
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)

        # Data augmentation
        if self.transforms:

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
        if self.transforms and random.random() > 0.5:
            trancolor = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            img = trancolor(img)

        img = np.array(img)

        # Load mask
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)

        # Data augmentation
        if self.transforms:

            # Random horizontal flipping
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

        # Convert mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask != 0] = 1

        # To tensor
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))

        return img, mask


class DetectNetDataset(Dataset):

    def __init__(self, root: str, mode: str = 'train', transforms: bool = False):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.resize = T.Resize(size=(512, 512))
        self.ddepth = cv2.CV_16S

        if mode == 'train':
            # TODO: add synthetic data
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
        else:
            raise NotImplementedError()

    def laplacian_filter(self, img, reps: int):
        ret = img.copy()
        for i in range(reps):
            ret = cv2.Laplacian(ret, self.ddepth, ksize=3)
            ret = cv2.convertScaleAbs(ret)
        return ret

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        # Apply color transformation
        if self.transforms and random.random() > 0.5:
            trancolor = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5)
            img = trancolor(img)

        img = np.array(img)

        # Load mask
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)

        # Data augmentation
        if self.transforms:

            # Random horizontal flipping
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

        # Convert mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask != 0] = 1

        # Resize
        img = np.array(self.resize(Image.fromarray(img)))
        mask = np.array(self.resize(Image.fromarray(mask)))

        # Sobel x and y
        sobel_x = cv2.Sobel(img, self.ddepth, 1, 0)
        sobel_x = cv2.convertScaleAbs(sobel_x)

        sobel_y = cv2.Sobel(img, self.ddepth, 0, 1)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        # Laplacian
        laplace_4 = self.laplacian_filter(img, 2)

        laplace_8 = self.laplacian_filter(img, 3)

        # To tensor
        img = np.transpose(img, (2, 0, 1))
        sobel_x = np.transpose(sobel_x, (2, 0, 1))
        sobel_y = np.transpose(sobel_y, (2, 0, 1))
        laplace_4 = np.transpose(laplace_4, (2, 0, 1))
        laplace_8 = np.transpose(laplace_8, (2, 0, 1))

        img = torch.from_numpy(img.astype(np.float32))
        sobel_x = torch.from_numpy(sobel_x.astype(np.float32))
        sobel_y = torch.from_numpy(sobel_y.astype(np.float32))
        laplace_4 = torch.from_numpy(laplace_4.astype(np.float32))
        laplace_8 = torch.from_numpy(laplace_8.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))

        # return img, sobel_x, sobel_y, laplace_4, laplace_8, mask
        return torch.cat([img, sobel_x, sobel_y, laplace_4, laplace_8], dim=0), mask


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