"""
Faster R-CNN
"""


import torch
import numpy as np
from torch.nn import functional as F
from torchvision.ops import nms
from vision.faster_rcnn.dataset import preprocess
from vision.faster_rcnn.utils import *


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f


class FasterRCNN(torch.nn.Module):
    """Base class for Faster R-CNN.
    This is a base class for Faster R-CNN links supporting object detection API.
    The following 3 stages consitute Faster R-CNN:
        1. Feature extraction: Images are taken and their feature maps are
            calculated.
        2. Region proposal networks: Given the feature maps calculated in the
            previous stage, produce set of RoIs around objects.
        3. Localization and classification heads: Using feature maps that belong
            to the proposed RoIs, classify the categories of the object in the RoIs
            and improve localizations.

    Args:
        extractor (torch.nn.Module): A module that takes a BxCxHxW image array and
            returns feature maps.
        rpn (torch.nn.Module): Region proposal network. 
        head (torch.nn.Module): A module that takes a BxCxHxW variable, RoIs, and batch
            indices for RoIs. This returns class dependent localization parameters and
            class scores.
        loc_norm_mean (tuple, optional): Mean values of localization estimates.
            Defaults to (0.0, 0.0, 0.0, 0.2).
        loc_norm_std (tuple, optional): Standard deviation of localization estimates.
            Defaults to (0.1, 0.1, 0.2, 0.2).
    """

    def __init__(self, extractor, rpn, head,
                 loc_norm_mean=(0.0, 0.0, 0.0, 0.2),
                 loc_norm_std=(0.1, 0.1, 0.2, 0.2)):

        super(FasterRCNN, self).__init__()

        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.loc_norm_mean = loc_norm_mean
        self.loc_norm_std = loc_norm_std
        self.use_preset('eval')

    @property
    def n_class(self):
        """Total number of classes including the background.

        Returns:
            int: Number of classes.
        """
        return self.head.n_class

    def forward(self, x, scale: float = 1.0):
        """Forward Faster R-CNN.

        Args:
            x (torch.autograd.Variable): 4D image variable.
            scale (float, optional): Amount of scaling applied to the raw image during preprocessing. Defaults to 1.0.

        Returns:
            Variable, Variable, array, array: roi_cls_locs - offsets and scalings
            for the proposed RoIs. roi_scores - class predictions for the proposed
            RoIs. rois - RoIs proposed by RPN. roi_indices - batch indices of RoIs.
        """
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset: str):
        """Use the given preset during prediction.
        This method changes values of nms_thresh and score_thresh. These values
        are a threshold value used for non maximum suppression and a threshold
        value to discard low confidence proposals in predict, respectively.

        Args:
            preset (str): A string to determine the preset to use.

        Raises:
            ValueError: Preset must be vis or eval.
        """
        if preset == 'vis':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'eval':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError(f'preset must be vis or eval, got {preset}')

    def _suppress(self, raw_cls_bbox, raw_prob):
        """[summary]

        Args:
            raw_cls_bbox ([type]): [description]
            raw_prob ([type]): [description]

        Returns:
            [type]: bounding box, label, score
        """
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class): # Skip cls_id = 0 because it is backgrounn class.
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),))) # The labels are in [0, self.n_class - 2].
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()

        if visualize:
            self.use_preset('vis')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(to_numpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes, labels, scores = list(), list(), list()
        for img, size in zip(prepared_imgs, sizes):
            img = to_tensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)

            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = to_tensor(rois) / scale

            mean = torch.Tensor(self.loc_norm_mean).cuda().repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_norm_std).cuda().repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)

            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)

            cls_bbox = loc2bbox(to_numpy(roi).reshape((-1, 4)),
                                to_numoy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = to_tensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(to_tensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        
        self.use_preset('eval')
        self.train()

        return bboxes, labels, scores
