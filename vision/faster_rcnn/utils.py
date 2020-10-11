"""
Utils for Faster R-CNN.
"""


import time
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.ops import nms


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    return data


def to_tensor(data, cuda: bool = True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)

    if isinstance(data, torch.Tensor):
        tensor = data.detach()

    if cuda:
        tensor = tensor.cuda()

    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        data = data.reshape(1)[0]

    if isinstance(data, torch.Tensor):
        data = data.item()

    return data


def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales."

    Args:
        src_bbox (array): A coordinates of bounding box.
        loc (array): An array with offsets and scales.

    Returns:
        array: Decoded bounding box coordiantes.
    """

    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]

    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding box to "loc".

    Args:
        src_bbox (array): An image coordinate array.
        dst_bbox (array): An image coordinate array.

    Returns:
        array: Bounding box offsets and scales.
    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()

    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection and area of the union.

    Args:
        bbox_a ([type]): An array.
        bbox_b (array): An array.

    Raises:
        IndexError: [description]

    Returns:
        array: An array.
    """

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size: int = 16, ratios=[0.5, 1.0, 2.0],
                         anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.
    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    Args:
        base_size (int, optional): The width and height of the reference window.
        Defaults to 16.
        ratios (list, optional): This is ratios of width to height of the anchors.
        Defaults to [0.5, 1.0, 2.0].
        anchor_scales (list, optional): This is areas of anchors. Those areas will
        be the product of the square of an element in anchor_scales and the original
        area of the reference window. Defaults to [8, 16, 32].

    Returns:
        numpy.ndarray: An array of shape (R, 4). Each element is a set of coordinates
        of a bounding box. The second axis corresponds to (y_min, x_min, y_max, x_max)
        of a bounding box.
    """

    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (numpy.ndarray): An array of shape (3, height, width). This is in RGB
            format and the range of its value is [0, 255].
        ax (matplotlib.axes.Axis, optional): The visualization is displayed on this
            axis. If None, a new axis is created. Defaults to None.

    Returns:
        matplotlib.axes.Axis: The Axes object with the plot for further tweaking. 
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))

    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    """Visualize bounding boxes inside image.

    Args:
        img (numpy.ndarray): An array of shape (3, height, width). This is in RGB
            format and the range of its value is [0, 255].
        bbox (numpy.ndarray): An array of shape (R, 4), where each element is
            organized by (y_min, x_min, y_max, x_max) in the second axis.
        label (numpy.ndarray, optional): An integer array of shape (R, ). The values
            correspond to id for label names. Defaults to None.
        score (numpy.ndarray, optional): A float array of shape (R, ). Each value
            indicates how confident the prediction is. Defaults to None.
        ax (matplot.axes.Axis, optional): The visualization is diplayed on this axis.
            If None, a new axis is created. Defaults to None.

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]

    Returns:
        matplotlib.axes.Axis: The Axes object with the plot for further tweaking.
    """

    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']

    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])

        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    return ax


class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    Args:
        n_sample (int, optional): The number of sampled regions. Defaults to 128.
        pos_ratio (float, optional): Fraction of regions that is labeled as a
            foreground. Defaults to 0.25.
        pos_iou_thresh (float, optional): IoU threshold for a RoI to be considered
            as a foreground. Defaults to 0.5.
        neg_iou_thresh_hi (float, optional): RoI is considered to be the background
            if IoU is in [neg_iou_thresh_lo, neg_iou_thresh_hi]. Defaults to 0.5.
        neg_iou_thresh_lo (float, optional): See neg_iou_thresh_hi. Defaults to 0.0.
    """

    def __init__(self, n_sample: int = 128, pos_ratio: float = 0.25,
                 pos_iou_thresh: float = 0.5, neg_iou_thresh_hi: float = 0.5,
                 neg_iou_thresh_lo: float = 0.0):
        
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_norm_mean=(0.0, 0.0, 0.0, 0.0),
                 loc_norm_std=(0.1, 0.1, 0.2, 0.2)):

        """Assigns ground truth to sampled proposals.
        This functions sampels total of self.n_sample RoIs from the combination of
        roi and bbox. The RoIs are assigned with the ground truth class labels as
        well as bounding box offsets and scales to match the ground truth bounding
        box. As many as pos_ratio * self.n_sample RoIs are sampled as foregrounds.

        Args:
            roi (array): Region of Interest (RoIs) from which we sample. Its shape
                is (R, 4).
            bbox (array): The coordinates of ground truth bounding boxes. its shape
                is (R, 4).
            label (array): Ground truth bounding box labels. Its shape (R, ). Its range
                is [0, L - 1], where L is the number of foreground classes.
            loc_norm_mean (tuple, optional): Mean values to normalize coordinates of
            bounding boxes. Defaults to (0.0, 0.0, 0.0, 0.0).
            loc_norm_std (tuple, optional): Standard deviation of the coordinates of
            bounding boxes. Defaults to (0.1, 0.1, 0.2, 0.2).

        Returns:
            array, array, array:
                sample_roi - regions of interests taht are sampled.
                    its shape is (S, 4).
                gt_roi_loc - offsets and scales to match the
                    sampled RoIs to the ground truth bounding boxes. Its shape is (S, 4).
                gt_roi_label - labels assigned to sampled RoIs. Its shape is (S, ).
                    Its range [0, L]. The label with value 0 is the background.
        """
        
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image,
                                         replace=False)

        # Select background RoIs as those within [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                              (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image,
                                         replace=False)

        # The indices that we are selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0 # Negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_norm_mean, np.float32)) / np.array(loc_norm_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.
    
    Assigns the ground truth bounding boxes to anchors for training Region Proposal
    Networks introduced in Faster R-CNN.

    Offsets and scales to match anchors to the ground truth are calculated using
    the endonding scheme of bbox2loc.

    Args:
        n_sample (int, optional): The number of regions to produce. Defaults to 256.
        pos_iou_thresh (float, optional): Anchors with IoU above this threshold will
            be assigned as positive. Defaults to 0.7.
        neg_iou_thresh (float, optional): Anchors with IoU below this threshold will
            be assigned as negative. Defaults to 0.3.
        pos_ratio (float, optional): Ratio of positive regions in the sampeld regions.
            Defaults to 0.5.
    """

    def __init__(self, n_sample: int = 256, pos_iou_thresh: float = 0.7,
                 neg_iou_thresh: float = 0.3, pos_ratio: float = 0.5):
        
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is (R, 4).
            anchor (array): Coordinates of anchors. Its shape is (S, 4).
            img_size (tuple(int, int)): A tuple of H and W, which is a tuple of height
                and width of an image. 

        Returns:
            array, array:
                loc - offsets and scales to match the anchors to the ground
                    truth bounding boxes. Its shape is (S, 4).
                label - labels of anchors with values (1=positive, 0=negative,
                    -1=ignore). Its shape is (S, ).
        """

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):

        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):

        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill: int = 0):
    """Unmap a subset of item (data) back to the original set of items
    (of size count).

    Args:
        data ([type]): [description]
        count ([type]): [description]
        index ([type]): [description]
        fill (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data

    return ret


def _get_inside_index(anchor, H, W):
    """Calculate indicies of anchors which are located completely inside of the image
    whose size is speficied.

    Args:
        anchor ([type]): [description]
        H ([type]): [description]
        W ([type]): [description]

    Returns:
        [type]: [description]
    """

    index_inside = np.where((anchor[:, 0] >= 0) &
                            (anchor[:, 1] >= 0) &
                            (anchor[:, 2] <= H) &
                            (anchor[:, 3] <= W))[0]

    return index_inside


class ProposalCreator:

    def __init__(self, parent_model, nms_thresh: float = 0.7,
                 n_train_pre_nms: int = 12000, n_train_post_nms: int = 2000,
                 n_test_pre_nms: int = 6000, n_test_post_nms: int = 300,
                 min_size: int = 16):
        
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale: float = 1.0):
        """Propose RoIs.

        Args:
            loc (array): Predicted offsets and scaling to anchors. Its shape is (R, 4).
            score (array): Predicted foreground probability for anchors. Its shape is
                (R, ).
            anchor (array): Coordinates of anchors. Its shape is (R, 4).
            img_size (tuple(int, int)): A tuple of height and width, which contains
                image size after scaling.
            scale (float, optional): The scaling factor used to scale an image after
                reading it from a file. Defaults to 1.0.

        Returns:
            array: An array of coordinates of proposal boxes. Its shape is (S, 4). S
                is less than self.n_test_post_nms in test time and less than
                self.n_train_post_nms in train time. S depends on the size of the
                predicted bounding boxes and the number of bounding boxes discarded
                by NMS.
        """

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = nms(torch.from_numpy(roi).cuda(),
                   torch.from_numpy(score).cuda(),
                   self.nms_thresh)

        if n_post_nms > 0:
            keep = keep[:n_post_nms]

        roi = roi[keep.cpu().numpy()]

        return roi