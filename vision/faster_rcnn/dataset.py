"""
Dataset for Faster R-CNN.
"""


import random
import torch
import numpy as np
import xml.etree.ElementTree as ET
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from torch.utils.data import Dataset
from PIL import Image


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


def read_image(path: str, dtype=np.float32, color: bool = True):
    """Read an image from a file.

    Args:
        path (str): A path of image file.
        dtype ([type], optional): The type of array. Defaults to np.float32.
        color (bool, optional): If True, the number of channels is 3, else 1. Defaults to True.

    Returns:
        numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding box according to image resize.

    The bounding box are expected to be packed into a two dimensional tensor of
    shape (R, 4), where R is the number of bounding boxes in the image. The
    second axis represents attributes of the bounding box. They are
    (y_min, x_min, y_max, x_max), where the 4 attributes are coordinates of
    the top left and the bottom right vertices.

    Args:
        bbox (numpy.ndarray): An array whose shape is (R, 4). R is the
            number of bounding boxes.
        in_size (tuple(int, int)): A tuple of length 2. The height and width of
            the image before resized.
        out_size (tuple(int, int)): A tuple of length 2. The height and width of
            the image after resized.

    Returns:
        numpy.ndarray: Bounding boxes according to the given image shapes.
    """

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip: bool = False, x_flip: bool = False):
    """Flip bounding boxes accordingly.

    Args:
        bbox (numpy.ndarray): An array whose shape is (R, 4). R is the number of bounding boxes.
        size (tuple(int, int)): A tuple of length 2. The height and width of the image before resized.
        y_flip (bool, optional): Flip bounding box according to a vertical flip of image. Defaults to False.
        x_flip (bool, optional): Flip bounding box according to a horizontal flip of image. Defaults to False.

    Returns:
        numpy.ndarray: Bounding box flipped according to the given flips.
    """

    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def crop_bbox(bbox, y_slice: slice = None, x_slice: slice = None,
              allow_outside_center: bool = True,
              return_param: bool = False):
    """Translate bounding boxes to fit within the cropped area of an image.

    Args:
        bbox (numpy.ndarray): Bounding boxes to be transformed. The shape is (R, 4). R is the number of bounding boxes.
        y_slice (slice, optional): The slice of y axis. Defaults to None.
        x_slice (slice, optional): The slice of x axis. Defaults to None.
        allow_outside_center (bool, optional): If False, bounding boxes whose centers are outside of the cropped area are remove. Defaults to True.
        return_param (bool, optional): If False, this function returns indices of kept bounding boxes. Defaults to False.

    Returns:
        numpy.ndarray: If return_param is False, returns an array of bbox.
        Else if return_param is True, returns a tuple whose elements are
        bbox and param. Param is a dictionary of intermediate parameters
        whose contents are listed below with key, value-type, and the
        description of the value.
            index: An array holding indices of used bounding boxes.
    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_: slice):
    """Slice to bounds

    Args:
        slice_ (slice):

    Returns:
        int, int: Lower and upper of slice.
    """

    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset: float = 0.0, x_offset: float = 0.0):
    """Translate bounding boxes. Defaults to

    Args:
        bbox (numpy.ndarray): Bounding boxes to be transformed.
        y_offset (float, optional): The offset along y axis. Defaults to 0.0.
        x_offset (float, optional): The offset along x axis. Defaults to 0.0.

    Returns:
        numpy.ndarray: Bounding boxes translated according to the given offsets.
    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_flip(img, y_random: bool = False, x_random: bool = False,
                return_param: bool = False, copy: bool = False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (numpy.ndarray): Image in CHW format.
        y_random (bool, optional): Randomly flip in vertical direction. Defaults to False.
        x_random (bool, optional): Randomly flip in horizontal direction. Defaults to False.
        return_param (bool, optional): Returns information of flip. Defaults to False.
        copy (bool, optional):  If False, a view of img will be returned. Defaults to False.

    Returns:
        numpy.ndarray or (numpy.ndarray, dict): 
    """

    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return 


class Transform(object):

    def __init__(self, min_size: int = 600, max_size: int = 1000):
        
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):

        img, bbox, label = data
        _, H, W = img.shape
        
        img = preprocess(img, self.min_size, self.max_size)
        _, oH, oW = img.shape
        scale = oH / H
        
        bbox = resize_bbox(bbox, (H, W), (oH, oW))

        img, params = random_flip(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (oH, oW), x_flip=params["x_flip"])

        return img, bbox, label, scale


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)


class VOCBboxDataset(Dataset):

    def __init__(self, root: str, mode: str = 'train', use_difficult: bool = False,
                 return_difficult: bool = False):

        self.root = root
        self.mode = mode
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        if mode == 'train':
            self.files = open(f'{root}/train.txt', 'rt').read().split('\n')[:-1]
            self.files = [os.path.join(root, f) for f in files]
        elif mode == 'val':
            self.files = open(f'{root}/val.txt', 'rt').read().split('\n')[:-1]
            self.files = [os.path.join(root, f) for f in files]
        elif mode == 'test':
            self.files = open(f'{root}/test.txt', 'rt').read().split('\n')[:-1]
            self.files = [os.path.join(root, f) for f in files]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        f = self.files[index]
        img = read_image(f'{f}.png', color=True)
        anno = ET.parse(f'{f}.xml')

        bbox, label, difficult = list(), list(), list()
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))

            bbox_anno = obj.find('bndbox')
            bbox.append([
                int(bbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax') 
            ])

            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        return img, bbox, label, difficult

    def get_example(self):
        index = random.randint(0, self.__len__())
        return self.__getitem__(index)
