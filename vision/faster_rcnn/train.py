"""
Trainer of the Faster R-CNN.
"""


import os
import sys
sys.path.append(os.getcwd())


import cv2
import json
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from vision.utils.detection.engine import train_one_epoch, evaluate
from vision.utils.detection import utils
from vision.utils.detection import transforms as T


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser('Faster R-CNN trainer')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/train_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='', help='path to save directory')
    parser.add_argument('--model_dir', type=str, default='vision/faster_rcnn/models', help='path to models directory')
    parser.add_argument('--resume_model', type=str, default='', help='path to model to resume')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to run')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for the dataloader')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = TexelDataset(args.data_dir, get_transform(train=True))
    val_data = TexelDataset(args.data_dir, get_transform(train=False))

    torch.manual_seed(1)
    indices = torch.randperm(len(train_data)).tolist()
    n_train = int(len(train_data) * 0.8)

    train_data = torch.utils.data.Subset(train_data, indices[-n_train:])
    val_data = torch.utils.data.Subset(val_data, indices[:-n_train])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=utils.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=utils.collate_fn    
    )

    n_classes = 2
    model = get_faster_rcnn(n_classes).to(device)

    if args.resume_model != '':
        wts = torch.load(args.resume_model)
        model.load_state_dict(wts)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    metric_collector = []

    for epoch in range(args.epochs):
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5)
        metric_collector.append(metric_logger)
        lr_scheduler.step()
        metric_logger_val = evaluate(model, val_loader, device)
        torch.save(model.state_dict(), f'{args.model_dir}/faster_rcnn_{args.epochs}.pt')

    for data in val_loader:
        img, target = data
        img = img[0]
        model.eval()
        with torch.no_grad():
            pred = model([img.to(device)])
        boxes = pred[0]['boxes'].cpu().numpy()
        labels = pred[0]['labels'].cpu().numpy()
        img = torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])(img)
        img = torchvision.transforms.ToPILImage()(img)
        img = np.array(img)
        visualize(img, boxes, labels)
        plt.show()
        plt.close('all')


def plot_metric_collector(data):
    loss, loss_clf, loss_box, loss_obj, loss_rpn = list(), list(), list(), list(), list()
    for d in data:
        loss.append(d.meters['loss'].avg)
        loss_clf.append(d.meters['loss_classifier'].avg)
        loss_box.append(d.meters['loss_box_reg'].avg)
        loss_obj.append(d.meters['loss_objectness'].avg)
        loss_rpn.append(d.meters['loss_rpn_box_reg'].avg)


def get_transform(train:bool):
    if train:
        transforms = A.Compose(
            [
                A.Resize(height=400, width=400, interpolation=cv2.INTER_CUBIC),
                A.ChannelShuffle(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Blur(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        transforms = A.Compose(
            [
                A.Resize(height=400, width=400, interpolation=cv2.INTER_CUBIC),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )
    return transforms


class TexelDataset(Dataset):
    def __init__(self, path:str, transforms=None):
        self.path = path
        self.transforms = transforms
        # self.image_dir = f'{path}/images'
        self.image_dir = f'{path}/labels'

        self.data = list()
        with open(f'{path}/annotations.json') as f:
            annotations = json.load(f)
    
            for key in annotations.keys():
                # if len(annotations[key][boxes]) == 0: break
                
                boxes, classes = list(), list()
                for anno in annotations[key]:
                    if 'timestamp' in anno.keys(): break

                    classes.append(anno['classId'])

                    xmin = int(anno['points']['x1'])
                    ymin = int(anno['points']['y1'])
                    xmax = int(anno['points']['x2'])
                    ymax = int(anno['points']['y2'])
                    boxes.append([xmin, ymin, xmax, ymax])
                
                if len(boxes) == 0: break
            
                img_name = key.replace('.png', '')
                data = {'image': img_name, 'classes': classes, 'boxes': boxes}
                self.data.append(data)
    
    def __len__(self): return len(self.data)

    def __getitem__(self, idx:int):
        img = self.data[idx]['image']
        # img = f'{self.image_dir}/{img}.jpg'
        img = f'{self.image_dir}/{img}.png'
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = self.data[idx]['classes']
        boxes = self.data[idx]['boxes']

        rows, cols, _ = img.shape
        for box in boxes:
            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[2] > cols: box[2] = cols
            if box[3] > rows: box[3] = rows

        if self.transforms is not None:
            aug = self.transforms(image=img, bboxes=boxes, category_ids=labels)
            img, boxes = aug['image'], aug['bboxes']

        n_box = len(boxes)
        if n_box > 0: boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else: boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((n_box,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return img, target


def get_faster_rcnn(n_classes:int):
    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    anchor_generator = AnchorGenerator(
        sizes=tuple([(16, 32, 64, 128, 256) for _ in range(5)]),
        aspect_ratios=tuple([(0.75, 0.5, 1.25) for _ in range(5)])
    )
    
    rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    
    faster_rcnn.rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=48,
        positive_fraction=0.5,
        pre_nms_top_n=dict(training=200, testing=100),
        post_nms_top_n=dict(training=160, testing=80),
        nms_thresh=0.7
    )

    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    faster_rcnn.roi_heads.fg_bg_sampler.batch_size_per_image = 24
    faster_rcnn.roi_heads.fg_bg_sampler.positive_fraction = 0.5

    return faster_rcnn


def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (255, 0, 0), -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name:dict={0: 'background', 1: 'fence'}):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    # plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img)


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')

    # dataset = TexelDataset(args.data_dir, get_transform(train=False))
    # for data in dataset:
    #     img, target = data
    #     img = torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])(img)
    #     img = torchvision.transforms.ToPILImage()(img)
    #     img = np.array(img)
    #     boxes = target['boxes']
    #     category_ids = [1 for _ in range(len(boxes))]
    #     visualize(img, boxes, category_ids)
    #     plt.show()
    #     plt.close('all')
