"""
Helpers for SegNet training and testing
"""


import os
import sys
import numpy as np


def accuracy(label, pred):
    """calculate the accuracy of a prediction

    Args:
        pred (np.array): prediction
        label (np.array): ground truth label

    Returns:
        float, int: accuracy, number of pixel
    """
    valid = (pred >= 0.05)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersection_over_union(label, pred, num_classes:int=1):
    """calculate intersection over union of a prediction

    Args:
        pred (np.array): prediction
        label (np.array): ground truth label
        num_classes (int): number of classes

    Returns:
        flot: intersection and union
    """
    pred = np.asarray(pred).copy()
    label = np.asarray(label).copy()
    pred += 1
    label += 1
    pred = pred * (label > 0)
    intersection = pred * (pred == label)
    area_intersection, _ = np.histogram(intersection, bins=num_classes, range=(1, num_classes))
    area_pred, _ = np.histogram(pred, bins=num_classes, range=(1, num_classes))
    area_label, _ = np.histogram(label, bins=num_classes, range=(1, num_classes))
    area_union = area_pred + area_label - area_intersection
    return area_intersection, area_union


def get_best_loss(path: str):
    assert os.path.exists(os.path.abspath(path)), 'path does not exists'

    all_files = os.listdir(os.path.abspath(path))
    files = list(filter(lambda file: file.endswith('.pth'), all_files))
    files = [s.replace('.pth', '') for s in files]

    min_val = np.inf
    for f in files:
        data = f.split('_')
        if len(data) > 2:
            val = float(data[2])
            if val < min_val:
                min_val = val

    return min_val
