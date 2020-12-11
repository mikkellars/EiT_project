import os
import sys
sys.path.append(os.getcwd())

import cv2
import random
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm

from vision.utils.split_data import split_data


n_images = 500

bg_dir = '/home/mathias/Downloads/Linemod_preprocessed/data/01/rgb'
bgs = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]

mask_path = 'vision/create_fence/data/fence_mask.png'

img_dir = 'vision/create_fence/data/mask_fence/images'
mask_dir = 'vision/create_fence/data/mask_fence/labels'

transforms = A.Compose([
    A.ShiftScaleRotate(p=1.0),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Transpose(),
    A.RandomRotate90(),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.Blur(),
    A.CLAHE(),
    A.RandomGamma(p=1.0),
    A.GaussNoise(p=1.0)
])


def rand_color():
    color = list(np.random.choice(range(10, 30), size=3))
    return color


def create_image(bg, fg, mask):
    assert bg.shape == fg.shape
    assert len(mask.shape) == 2 

    bg = Image.fromarray(bg)
    fg = Image.fromarray(fg)
    mask = Image.fromarray(mask)

    bg.paste(fg, (0, 0), mask)

    return np.array(bg)


def rand_transform(im, mask):
    aug = transforms(image=im, mask=mask)
    im = aug['image']
    mask = aug['mask']
    return im, mask


if __name__ == '__main__':
    for i in tqdm(range(n_images), desc='Creating synthetic images of fence'):

        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        tmp = np.where(mask == 255, rand_color(), [0, 0, 0]).astype(np.uint8)
        tmp, mask = rand_transform(tmp, mask)

        h, w, c = tmp.shape

        bg = random.choice(bgs)
        bg = cv2.imread(bg, cv2.IMREAD_COLOR)
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

        tmp = create_image(bg, tmp, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        cv2.imwrite(f'{img_dir}/{i:03d}.png', tmp)
        cv2.imwrite(f'{mask_dir}/{i:03d}.png', mask)

    split_data(
        input_dir='vision/create_fence/data/mask_fence/images',
        output_dir='vision/create_fence/data/mask_fence',
        split_train=True
    )