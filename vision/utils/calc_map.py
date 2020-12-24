"""
Model validations metrics
"""


from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os


# def xywh2xyxy(x:np.array):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y = np.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y

def xywh2xyxy(x:np.array):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

def ap_per_class(tp, conf, pred_cls, target_cls, plot:bool=False,
                 save:str='precision-recall_curve.png',
                 names:list=[]):
    """
    Compute the average precision, given the recall and precision curves.
    Src: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments:
        tp: True positives (np.array).
        conf: Objectness value from 0-1 (np.array).
        pred_cls: Predicted object classes (np.array).
        target_cls: True object classes (np.array).
        plot: Plot precision curve at mAP@0.5 (bool, default false).
        save: Save path of plot (str, default 'precision-recall_curve.png').
    # Returns:
        The average precision as computed in py-faster-rcnn.
    """
    
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_cls = np.unique(target_cls)
    
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []
    pr_score = 0.1
    s = [unique_cls.shape[0], tp.shape[1]]
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_cls):
        i = pred_cls == c
        num_labels = (target_cls == c).sum() # Number of labels
        num_preds = i.sum() # Number of predictions
        
        if num_preds == 0 or num_labels == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (num_labels + 1e-16)
        r[ci] = np.interp(-pr_score, -conf[i], recall[:,0])
        
        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-pr_score, -conf[i], precision[:,0])

        # AP from precision-recall curve
        for j in range(tp.shape[1]):
            ap[ci,j], mpre, mrec = compute_ap(recall[:,j], precision[:,j])
            if plot and (j==0):
                py.append(np.interp(px, mrec, mpre))
    
    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    
    if plot:
        plot_pr_curve(px, py, ap, save, names)
    
    return p, r, ap, f1, unique_cls.astype('int32')


def compute_ap(recall:list, precision:list):
    """
    Compute average precision, given the recall and precision curves.
    Src: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments:
        recall: The recall curve (list).
        precision: The precision curve (list).
    # Returns:
        The average precision as compute in py-faster-rcnn.
    """
    
    mrec = recall
    mpre = precision
    
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # Integrate area under curve
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    
    return ap, mpre, mrec


def box_iou(box1, box2):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])

    # compute the area of intersection rectangle
    inter = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter / float(area1 + area2 - inter + 1e-16)

    # return the intersection over union value
    return np.array([iou])


class ConfusionMatrix:
    
    def __init__(self, num_cls, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((num_cls+1, num_cls+1))
        self.num_cls = num_cls
        self.conf = conf
        self.iou_thres = iou_thres
        
    def process_batch(self, detections, labels):
        """Return intersection over union (Jaccard index) of boxes.
        Both sets of boces are expected to bin in (x1, y1, x2, y2) format.

        Args:
            detections (np.array): x1, y1, x2, y2, conf, class
            labels (np.array): class, x1, y1, x2, y2
        """
        detections = detections[detections[:,4] > self.conf]
        gt_classes = labels[:,0].astype(np.int32)
        detection_classes = detections[:,5].astype(np.int32)
        iou = box_iou(labels[:,1:], detections[:,:4])

        x = np.where(iou > self.iou_thres)
        if x[0].shape[0]:
            # matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
            matches = np.concatenate((np.stack(x, 1), iou), 1)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[gc, self.num_cls] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[self.num_cls, dc] += 1  # background FN

    def matrix(self): return self.matrix

    def plot(self, save_dir:str='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9))
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FN'] if labels else "auto",
                       yticklabels=names + ['background FP'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.tight_layout()
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)

        except Exception as e:
            print('Could not import seaborn, try to install it.')
            pass

    def print(self):
        for i in range(self.nc+1):
            print(f'{map(str, self.matrix[i])}')


def plot_pr_curve(px, py, ap, save:str='', names=()):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # show mAP in legend if < 10 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} %.3f' % ap[i, 0])  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / 'precision_recall_curve.png', dpi=250)


if __name__ == '__main__':
    print(__doc__)
    start_time = time.time()

    # gt_folder = '/home/mathias/Documents/experts_in_teams_proj/vision/yolov5/data/fence_test_sim/test/labels'
    # gts = sorted([f for f in os.listdir(gt_folder)])

    # preds_folder = '/home/mathias/Documents/experts_in_teams_proj/vision/yolov5/runs/detect/sim/labels'
    # preds = sorted([f.split('.')[0] for f in os.listdir(preds_folder)])

    # confusion_mat = ConfusionMatrix(num_cls=1, conf=0.25, iou_thres=0.45)

    # for i in tqdm(range(0, len(gts))):
    #     label = gts[i]

    #     try: idx = preds.index(label.split('_')[0])
    #     except: idx = None

    #     # Labels
    #     lines = open(os.path.join(gt_folder, label), 'r').readlines()
    #     if len(lines) == 0:
    #         continue
    #     else:
    #         gt = np.zeros((len(lines), 5))
    #         for i, l in enumerate(lines):
    #             l = l.replace('\n', '').split(' ')
    #             gt[i, 0] = int(l[0])
    #             gt[i, 1] = float(l[1])
    #             gt[i, 2] = float(l[2])
    #             gt[i, 3] = float(l[3])
    #             gt[i, 4] = float(l[4])

    #     # Predictions
    #     if idx is not None:
    #         pred = preds[idx]
    #         lines = open(os.path.join(preds_folder, pred+'.txt'), 'r').readlines()
    #         if len(lines) > 0:
    #             p = np.zeros((len(lines), 6))
    #             for i, l in enumerate(lines):
    #                 l = l.replace('\n', '').split(' ')
    #                 p[i, 0] = float(l[1])
    #                 p[i, 1] = float(l[2])
    #                 p[i, 2] = float(l[3])
    #                 p[i, 3] = float(l[4])
    #                 p[i, 4] = float(l[5])
    #                 p[i, 5] = int(l[0])
    #     else:
    #         pass

    #     p[:, 0:4], gt[:, 1:5] = xywh2xyxy(p[:, 0:4]), xywh2xyxy(gt[:, 1:5])
    #     confusion_mat.process_batch(detections=p, labels=gt)

    # confusion_mat.plot()

    # Load labels into ground-truth folder
    gt = '/home/mathias/Documents/experts_in_teams_proj/vision/yolov5/data/fence_test_real/test/labels'
    gt_files = sorted([f for f in os.listdir(gt)])
    preds = '/home/mathias/Documents/experts_in_teams_proj/vision/yolov5/runs/detect/real_airport/labels'
    preds_files = sorted([f for f in os.listdir(preds)])
    for i, f in enumerate(gt_files):
        with open(os.path.join(gt, f), 'r') as txt_file:
            lines = txt_file.readlines()
        with open(f'/home/mathias/Documents/experts_in_teams_proj/vision/utils/map/input/ground-truth/{i:04d}.txt', 'w') as txt_file:
            if len(lines) == 0:
                    # print('', file=txt_file)
                    continue
            else:
                for line in lines:
                    line = line.replace('\n', '').split(' ')
                    bbox = xywh2xyxy(np.array([
                        float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    ]))
                    print(f'hole {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}', file=txt_file)


        with open(f'/home/mathias/Documents/experts_in_teams_proj/vision/utils/map/input/detection-results/{i:04d}.txt', 'w') as txt_file:
            try:
                idx = f.split('_')[0]
                idx = preds_files.index(f'{idx}.txt')
                pred = preds_files[idx]
                with open(os.path.join(preds, pred), 'r') as input_txt:
                    lines = input_txt.readlines()
                if len(lines) == 0:
                        # print('', file=txt_file)
                        continue
                else:
                    for line in lines:
                        line = line.replace('\n', '').split(' ')
                        bbox = xywh2xyxy(np.array([
                            float(line[1]), float(line[2]), float(line[3]), float(line[4])
                        ]))
                        print(f'hole {line[5]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}', file=txt_file)
            except:
                # print('', file=txt_file)
                continue

    end_time = time.time() - start_time
    print(f'It took {end_time//60} minutes and {end_time%60:.0f} seconds.')