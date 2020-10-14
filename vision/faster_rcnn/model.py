"""
Faster R-CNN
"""


import os
import sys
sys.path.append(os.getcwd())


import torch
import numpy as np
from torch.nn import functional as F
from torchvision.ops import nms, RoIPool
from torchvision.models import vgg16
from vision.faster_rcnn.dataset import preprocess
from vision.faster_rcnn.helpers import *


# ------------
# Faster R-CNN
# ------------


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


# -------------------------
# Regional Proposal Network
# -------------------------


class RegionalProposalNetwork(torch.nn.Module):

    def __init__(self, in_channels: int = 512, mid_channels: int = 512,
                 ratios=[0.5, 1.0, 2.0], anchor_scales=[8, 16, 32],
                 feat_stride: int = 16, proposal_creator_params=dict()):
        
        super(RegionalProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales,
                                                ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = torch.nn.Conv2d(mid_channels, n_anchor*2, 1, 1, 0)
        self.loc = torch.nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale: float = 1.0):
        """Forward Region Proposal Network.

        Args:
            x (torch.autograd.Variable): The features extracted from images. Its shape is (N, C, H, W).
            img_size (tuple(int, int)): A tuple of height and width, which contains image size after scaling.
            scale (float, optional): The amount of scaling done the the input images after reading the from files. Defaults to 1.0.

        Returns:
            torch.autograd.Variable, torch.autograd.Variable, array, array, array:
                rpn_locs - predicted bounding box offsets and scales for anchors. Its shape is (N, H, W, A, 4).
                rpn_scores - predicted foreground scores for anchors. Its shape is (N, H, W, A, 2).
                rois - A bounding box array containing coordinates of proposal boxes. This is concatenation of bounding box
                    arrays from multiple images in the batch. Its shape is (R, 4).
                roi_indices - an array containing indices of images to whcih RoIs correspond to. Its shape is (R, ).
                anchor - coordinates of enumerated shifted anchors. Its shape is (H, W, A, 4).
        """

        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(),
                                      rpn_fg_scores[i].cpu().data.numpy(),
                                      anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, std, truncated: bool = False):
    """Weight initalizer: truncated normal and random normal.

    Args:
        m ([type]): [description]
        mean ([type]): Mean.
        std ([type]): Standard deviation.
        truncated (bool, optional): [description]. Defaults to False.
    """

    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(std).add_(mean)
    else:
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """Enumerate all shifted anchors:
        Add A anchors (1, A, 4) to cell K shifts (K, 1, 4) to get shift
        anchors (K, A, 4).
        Reshape to (K*A, 4) shifted anchors.
        Return anchor (K*A, 4).

    Args:
        anchor_base ([type]): Anchor base.
        feat_stride ([type]): Feature stride.
        height ([type]): Height.
        width ([type]): Width.

    Returns:
        [type]: [description]
    """

    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)

    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]

    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    """Enumerate all shifted anchors:
        Add A anchors (1, A, 4) to cell K shifts (K, 1, 4) to get shift
        anchors (K, A, 4).
        Reshape to (K*A, 4) shifted anchors.
        Return anchor (K*A, 4).

    Args:
        anchor_base ([type]): Anchor base.
        feat_stride ([type]): Feature stride.
        height ([type]): Height.
        width ([type]): Width.

    Returns:
        [type]: [description]
    """

    shift_y = torch.arange(0, height*feat_stride, feat_stride)
    shift_x = torch.arange(0, width*feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]

    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K*A, 4)).astype(np.float32)

    return anchor


# ------------------
# Faster R-CNN VGG16
# ------------------


def decom_vgg16(pretrained: bool = True, use_drop: bool = True,
                freeze_feat: int = 30):

    assert freeze_feat <= 30, f'freeze_feat must not be greater than 30, got {freeze_feat}'

    model = vgg16(pretrained=True)

    extractor = list(model.extractor)[:freeze_feat]
    for layer in extractor:
        for p in layer.parameters():
            p.requires_grad = False
    extractor = torch.nn.Sequential(*extractor)

    classifier = list(model.classifier)
    del classifier[6]
    if use_drop is False:
        del classifier[5]
        del classifier[2]
    classifier = torch.nn.Sequential(*classifier)

    return extractor, classifier


class VGG16RoIHead(torch.nn.Module):

    def __init__(self, n_class: int, roi_size: int, spatial_scale: float,
                 classifier: torch.nn.Module):

        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = torch.nn.Linear(4096, n_class*4)
        self.score = torch.nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.1)
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x: torch.autograd.Variable, rois: torch.Tensor,
                roi_indices: torch.Tensor):
        """Forward the chain.

        Args:
            x (torch.autograd.Variable): 4D image variable.
            rois (torch.Tensor): A bounding box array containing coordinates of proposal boxes.
                This is a concatenation of bounding box arrays from multiple images in the batch.
                Its shape is (R, 4).
            roi_indices (torch.Tensor): An array containing indices of images to which bounding boxes
                correspond to. Its shape is (R, ).

        Returns:
            array, array: roi_cls_locs, roi_scores.
        """

        roi_indices = to_tensor(roi_indices).float()
        rois = to_tensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


class FasterRCNNVGG16(FasterRCNN):

    feat_stride = 16 # Downsample 16x for output of conv5 in vgg16

    def __init__(self, n_fg_class: int = 20, ratios=[0.5, 1.0, 2.0],
                 anchor_scales=[8, 16, 32]):
        """[summary]

        Args:
            n_fg_class (int, optional): [description]. Defaults to 20.
            ratios (list, optional): [description]. Defaults to [0.5, 1.0, 2.0].
            anchor_scales (list, optional): [description]. Defaults to [8, 16, 32].
        """

        extractor, classifier = decom_vgg16()
        rpn = RegionalProposalNetwork(512, 512, ratios=ratios,
                                      anchor_scales=anchor_scales,
                                      feat_stride=self.feat_stride)
        head = VGG16RoIHead(n_class=n_fg_class+1, roi_size=7,
                            spatial_scale=(1.0/self.feat_stride),
                            classifier=classifier)
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)