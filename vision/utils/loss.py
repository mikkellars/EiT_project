


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# def cross_entropy2d(inputs, targets, weight=None, size_average=True):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = targets.size()
#     if h != ht and w != wt: inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)
#     inputs = inputs.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
#     targets = targets.view(-1)
#     loss = F.cross_entropy(inputs, targets, weight=weight, size_average=size_average, ignore_index=250)
#     return loss

class CrossEntropy2d(_Loss):
    """Cross Entropy 2d loss

    Args:
        weight (torch.Tensor, optional): Weight. Defaults to None.
        size_average (bool, optional): Size average. Defaults to True.
    """

    def __init__(self, weight:torch.Tensor=None, size_average:bool=True):
        super(CrossEntropy2d, self).__init__(True)
        self.weight = weight
        self.size_average = size_average
    
    def forward(self, x, target):
        n, c, h, w = x.size()
        nt, ct, ht, wt = target.size()
        if h != ht and w != wt: x = F.interpolate(x, size=(ht, wt), mode='bilinear', align_corners=True)
        x = x.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
        target = target.view(-1)
        loss = F.cross_entropy(x, target, weight=self.weight, size_average=self.size_average, ignore_index=250)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1.0 - dsc


class FocalLoss(nn.Module):

    def __init__(self, gamma:int=0, alpha=None, size_average:bool=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()