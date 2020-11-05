"""
Metrics
"""


import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


EPS = 1e-10


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice


class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self)->None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n:int=1)->None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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


class iouCalc():
    
    def __init__(self, classLabels, validClasses, voidClass = None):
        assert len(classLabels) == len(validClasses), 'Number of class ids and names must be equal'
        self.classLabels    = classLabels
        self.validClasses   = validClasses
        self.voidClass      = voidClass
        self.evalClasses    = [l for l in validClasses if l != voidClass]
        
        self.perImageStats  = []
        self.nbPixels       = 0
        self.confMatrix     = np.zeros(shape=(len(self.validClasses),len(self.validClasses)),dtype=np.ulonglong)
        
        # Init IoU log files
        self.headerStr = 'epoch, '
        for label in self.classLabels:
            if label.lower() != 'void':
                self.headerStr += label + ', '
            
    def clear(self):
        self.perImageStats  = []
        self.nbPixels       = 0
        self.confMatrix     = np.zeros(shape=(len(self.validClasses),len(self.validClasses)),dtype=np.ulonglong)
        
    def getIouScoreForLabel(self, label):
        # Calculate and return IOU score for a particular label (train_id)
        if label == self.voidClass:
            return float('nan')
    
        # the number of true positive pixels for this label
        # the entry on the diagonal of the confusion matrix
        tp = np.longlong(self.confMatrix[label,label])
    
        # the number of false negative pixels for this label
        # the row sum of the matching row in the confusion matrix
        # minus the diagonal entry
        fn = np.longlong(self.confMatrix[label,:].sum()) - tp
    
        # the number of false positive pixels for this labels
        # Only pixels that are not on a pixel with ground truth label that is ignored
        # The column sum of the corresponding column in the confusion matrix
        # without the ignored rows and without the actual label of interest
        notIgnored = [l for l in self.validClasses if not l == self.voidClass and not l==label]
        fp = np.longlong(self.confMatrix[notIgnored,label].sum())
    
        # the denominator of the IOU score
        denom = (tp + fp + fn)
        if denom == 0:
            return float('nan')
    
        # return IOU
        return float(tp) / denom
    
    def evaluateBatch(self, predictionBatch, groundTruthBatch):
        # Calculate IoU scores for single batch
        assert predictionBatch.size(0) == groundTruthBatch.size(0), 'Number of predictions and labels in batch disagree.'
        
        # Load batch to CPU and convert to numpy arrays
        predictionBatch = predictionBatch.cpu().numpy()
        groundTruthBatch = groundTruthBatch.cpu().numpy()

        for i in range(predictionBatch.shape[0]):
            predictionImg = predictionBatch[i,:,:]
            groundTruthImg = groundTruthBatch[i,:,:]
            
            # Check for equal image sizes
            assert predictionImg.shape == groundTruthImg.shape, 'Image shapes do not match.'
            assert len(predictionImg.shape) == 2, 'Predicted image has multiple channels.'
        
            imgWidth  = predictionImg.shape[0]
            imgHeight = predictionImg.shape[1]
            nbPixels  = imgWidth*imgHeight
            
            # Evaluate images
            encoding_value = max(groundTruthImg.max(), predictionImg.max()).astype(np.int32) + 1
            encoded = (groundTruthImg.astype(np.int32) * encoding_value) + predictionImg
        
            values, cnt = np.unique(encoded, return_counts=True)
        
            for value, c in zip(values, cnt):
                pred_id = value % encoding_value
                gt_id = int((value - pred_id)/encoding_value)
                if not gt_id in self.validClasses:
                    printError('Unknown label with id {:}'.format(gt_id))
                self.confMatrix[gt_id][pred_id] += c
        
            # Calculate pixel accuracy
            notIgnoredPixels = np.in1d(groundTruthImg, self.evalClasses, invert=True).reshape(groundTruthImg.shape)
            erroneousPixels = np.logical_and(notIgnoredPixels, (predictionImg != groundTruthImg))
            nbNotIgnoredPixels = np.count_nonzero(notIgnoredPixels)
            nbErroneousPixels = np.count_nonzero(erroneousPixels)
            self.perImageStats.append([nbNotIgnoredPixels, nbErroneousPixels])
            
            self.nbPixels += nbPixels
            
        return
            
    def outputScores(self):
        # Output scores over dataset
        assert self.confMatrix.sum() == self.nbPixels, 'Number of analyzed pixels and entries in confusion matrix disagree: confMatrix {}, pixels {}'.format(self.confMatrix.sum(),self.nbPixels)
    
        # Calculate IOU scores on class level from matrix
        classScoreList = []
            
        # Print class IOU scores
        outStr = 'classes           IoU\n'
        outStr += '---------------------\n'
        for c in self.evalClasses:
            iouScore = self.getIouScoreForLabel(c)
            classScoreList.append(iouScore)
            outStr += '{:<14}: {:>5.3f}\n'.format(self.classLabels[c], iouScore)
        miou = getScoreAverage(classScoreList)
        outStr += '---------------------\n'
        outStr += 'Mean IoU      : {avg:5.3f}\n'.format(avg=miou)
        outStr += '---------------------'
        
        print(outStr)
        
        return miou


def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)


def getScoreAverage(scoreList):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not np.isnan(score):
            validScores += 1
            scoreSum += score
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores


def plot_learning_curves(metrics, args):
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    ln5 = ax2.plot(x, metrics['miou'], color='tab:green')
    lns = ln1+ln2+ln3+ln4+ln5
    plt.legend(lns, ['Train loss','Validation loss','Train accuracy','Validation accuracy','mIoU'])
    plt.tight_layout()
    plt.savefig(args.save_path + '/learning_curve.png', bbox_inches='tight')


def visim(img, args):
    img = img.cpu()
    # Convert image data to visual representation
    img *= torch.tensor(args.dataset_std)[:,None,None]
    img += torch.tensor(args.dataset_mean)[:,None,None]
    npimg = (img.numpy()*255).astype('uint8')
    if len(npimg.shape) == 3 and npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    else:
        npimg = npimg[0,:,:]
    return npimg


def vislbl(label, mask_colors):
    label = label.cpu()
    # Convert label data to visual representation
    label = np.array(label.numpy())
    if label.shape[-1] == 1:
        label = label[:,:,0]
    
    # Convert train_ids to colors
    label = mask_colors[label]
    return label