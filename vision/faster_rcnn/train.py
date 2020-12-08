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
import albumentations as A
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from vision.utils.detection.engine import train_one_epoch, evaluate
from vision.utils.detection import utils


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser('Faster R-CNN trainer')
    parser.add_argument('--exp', type=str, default='hole', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/train_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='vision/faster_rcnn/images', help='path to save directory')
    parser.add_argument('--model_dir', type=str, default='vision/faster_rcnn/models', help='path to models directory')
    parser.add_argument('--resume_model', type=str, default='', help='path to model to resume')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to run')
    parser.add_argument('--bs', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for the dataloader')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = HoleDataset('vision/faster_rcnn/data/holes_in_fence_coco/train', get_transform(train=True))
    val_data = HoleDataset('vision/faster_rcnn/data/holes_in_fence_coco/valid', get_transform(train=False))
    test_data = HoleDataset('vision/faster_rcnn/data/holes_in_fence_coco/test', get_transform(train=False))

    # torch.manual_seed(1)
    # indices = torch.randperm(len(train_data)).tolist()
    # n_train = int(len(train_data) * 0.8)

    # train_data = torch.utils.data.Subset(train_data, indices[-n_train:])
    # val_data = torch.utils.data.Subset(val_data, indices[:-n_train])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=utils.collate_fn)

    n_classes = 2
    model = get_faster_rcnn(n_classes).to(device)

    if args.resume_model != '':
        wts = torch.load(args.resume_model)
        model.load_state_dict(wts)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    metric_collector_train = list()
    for epoch in range(args.epochs):
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5)
        metric_collector_train.append(metric_logger)
        # lr_scheduler.step()
        metric_logger_val = evaluate(model, val_loader, device)

    torch.save(model.state_dict(), f'{args.model_dir}/faster_rcnn_{args.exp}_{args.epochs}.pt')
    plot_metric_train(metric_collector_train, path=f'{args.save_dir}/hole/metric_logger_train.png')

    times = list()
    for i, data in enumerate(val_loader):
        img, target = data
        img = img[0]
        
        model.eval()
        with torch.no_grad():
            model_start = time.time()
            pred = model([img.to(device)])
            model_end = time.time() - start_time
        
        times.append(model_end)
        
        boxes = pred[0]['boxes'].cpu().numpy()
        labels = pred[0]['labels'].cpu().numpy()
        
        img = torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])(img)
        img = torchvision.transforms.ToPILImage()(img)
        img = np.array(img)

        path = f'{args.save_dir}/hole/val/{i:03d}.png'
        visualize(img, boxes, labels, path=path)

    for i, data in enumerate(test_loader):
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

        path = f'{args.save_dir}/hole/test/{i:03d}.png'
        visualize(img, boxes, labels, path=path)


def plot_metric_train(data, path=None):
    epochs = list()
    loss = list()
    loss_clf = list()
    loss_box = list()
    loss_obj = list()
    loss_rpn = list()

    for i, d in enumerate(data):
        epochs.append(i)
        loss.append(d.meters['loss'].avg)
        loss_clf.append(d.meters['loss_classifier'].avg)
        loss_box.append(d.meters['loss_box_reg'].avg)
        loss_obj.append(d.meters['loss_objectness'].avg)
        loss_rpn.append(d.meters['loss_rpn_box_reg'].avg)

    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='Loss')
    plt.plot(epochs, loss_clf, label='Loss classifier')
    plt.plot(epochs, loss_box, label='Loss box regression')
    plt.plot(epochs, loss_obj, label='Loss objectness')
    plt.plot(epochs, loss_rpn, label='Loss rpn box regression')
    plt.legend()

    if path is not None: plt.savefig(path)
    else: plt.show()

    plt.close('all')
    

def get_transform(train:bool, im_size:int=400):
    if train:
        transforms = A.Compose(
            [
                A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_CUBIC),
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
                A.Resize(height=im_size, width=im_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )
    return transforms


class TexelDataset(torch.utils.data.Dataset):
    def __init__(self, path:str, transforms=None):
        self.path = path
        self.transforms = transforms
        self.image_dir = f'{path}/images'
        # self.image_dir = f'{path}/labels'

        self.data = list()
        with open(f'{path}/annotations.json') as f:
            annotations = json.load(f)
    
            for key in annotations.keys():
                if key == 'Eskild_fig_3_16.jpg': continue
                
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
            
                # img_name = key.replace('.png', '')
                data = {'image': key, 'classes': classes, 'boxes': boxes}
                self.data.append(data)
    
    def __len__(self): return len(self.data)

    def __getitem__(self, idx:int):
        img = self.data[idx]['image']
        img = f'{self.image_dir}/{img}'
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


class HoleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(os.path.join(root, '_annotations.coco.json'))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        f = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, f), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_objs = len(coco_annotation)
        boxes, labels, areas = list(), list(), list()
        for i in range(num_objs):
            # Bbox
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            # label
            labels.append(coco_annotation[i]['category_id'])
            
            # area
            areas.append(coco_annotation[i]['area'])

        # labels = np.ones((num_objs, ), dtype=np.int64)

        # areas = []
        # for i in range(num_objs):
        #     areas.append(coco_annotation[i]['area'])

        if self.transforms is not None:
            aug = self.transforms(image=img, bboxes=boxes, category_ids=labels)
            img, boxes = aug['image'], aug['bboxes']

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        return img, my_annotation

    def __len__(self): return len(self.ids)


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


def visualize(image, bboxes, category_ids, path=None,
              category_id_to_name:dict={0: 'background', 1: 'Hole'}):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img)

    if path is not None: plt.savefig(path)
    else: plt.show()
    
    plt.close('all')


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')

    # # Show data
    # dataset = HoleDataset('vision/faster_rcnn/data/holes_in_fence_coco/train', get_transform(train=True))
    # print(f'Dataset has the length of {len(dataset)}')
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

    # # Show model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # n_classes = 2
    # model = get_faster_rcnn(n_classes).to(device)

    # wts = torch.load('vision/faster_rcnn/models/faster_rcnn_hole_200.pt')
    # model.load_state_dict(wts)