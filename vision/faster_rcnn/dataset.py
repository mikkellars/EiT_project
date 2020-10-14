"""
Dataset for Faster R-CNN.
"""


import torch
import numpy as np
from skimage import transform as sktsf
from torchvision import transforms as tvtsf


def normalize(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size: int = 600, max_size: int = 1000):
    """Proprocess an image for feature extraction.

    Args:
        img (numpy.ndarray): An image. This is in CxHxW and RGB format.
        min_size (int, optional): [description]. Defaults to 600.
        max_size (int, optional): [description]. Defaults to 1000.

    Returns:
        numpy.ndarray: A preprocessed image.
    """
    c, h, w = img.shape

    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)

    img = img / 255.0
    img = sktsf.resize(img, (c, h*scale, w*scale), mode='reflect', anti_aliasing=False)

    return normalize(img)