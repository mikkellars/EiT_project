"""
Instance segmentation with Mask R-CNN
"""


import os
import sys
sys.path.append(os.getcwd())

import cv2
import time
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from vision.utils.detection.engine import train_one_epoch, evaluate
from vision.utils.detection import utils
from vision.utils.detection import transforms as T


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Instance segmentation with Mask R-CNN')
    parser.add_argument('--name', type=str, default='segm_mask_rcnn_fence', help='Name of model in models')
    parser.add_argument('--train_dir', type=str, default='vision/data/fence_data/train_set', help='path to train data directory')
    parser.add_argument('--test_dir', type=str, default='vision/data/fence_data/test_set', help='path to test data directory')
    parser.add_argument('--models_dir', type=str, default='vision/models', help='path to models directory')
    parser.add_argument('--resume_model', type=str, default='vision/models/segm_mask_rcnn_fence_100.pt', help='path to model file')
    args = parser.parse_args()
    return args


def main(args):

    # ----------------------
    # Example of the dataset
    # ----------------------

    # dataset = PennFudanDataset('vision/data/PennFudanPed', get_transform(train=True))
    # dataset_val = PennFudanDataset(args.data_dir, get_transform(train=False))
    # for img, target in dataset_val: plot_image_w_mask(img, target)

    dataset = FenceDataset(args.train_dir, get_transform(train=True))
    dataset_val = FenceDataset(args.train_dir, get_transform(train=False))
    # for img, target in dataset_val: plot_image_w_mask(img, target)

    # -----------------------
    # Setup the dataloader class
    # -----------------------

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    n_train = int(len(dataset) * 0.8)
    dataset = torch.utils.data.Subset(dataset, indices[-n_train:])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[:-n_train])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8, collate_fn=utils.collate_fn)

    # -----------------------------
    # Setup the model and optimizer
    # -----------------------------

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    if args.resume_model != '':
        wts = torch.load(args.resume_model)
        model.load_state_dict(wts)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ---------------
    # Train the model
    # ---------------

    # num_epochs = 50
    # for epoch in range(num_epochs):
    #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    #     lr_scheduler.step()
    #     evaluate(model, data_loader_test, device=device)
    # torch.save(model.state_dict(), f'{args.models_dir}/{args.name}_100.pt')

    # ---------------
    # Show prediction
    # ---------------

    # lorenz_data = LorenzDataset('vision/data/lorenz-fence-inspection-examples', get_transform(train=False))
    # dataset_test = FenceDataset(args.test_dir, get_transform(train=False))

    # for i, data in enumerate(dataset_test):
    #     img, _ = data
    #     model.eval()
    #     with torch.no_grad():
    #         pred = model([img.to(device)])
    #     masks = torch.sigmoid(pred[0]['masks'])
    #     masks = (masks > 0.525).float()
    #     pred[0]['masks'] = masks
    #     plot_image_w_mask(img, pred[0])

    img_path = 'vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_16.jpg'
    # img_path = 'vision/data/lorenz-fence-inspection-examples/IMG_2353.jpg'
    # img_path = 'vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_17.jpg'
    patches = split_image_in_patches(img_path, 128)
    for img in patches:
        img = transforms.ToTensor()(img)
        model.eval()
        with torch.no_grad():
            pred = model([img.to(device)])
        masks = torch.sigmoid(pred[0]['masks'])
        masks = (masks > 0.51).float()
        pred[0]['masks'] = masks
        plot_image_w_mask(img, pred[0])


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))
    
    def __getitem__(self, idx:int):
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        mask = np.array(mask)
        mask = np.where(mask!=0, 1, 0)
        obj_ids = np.unique(mask)   # Instances are encoded as different colors
        obj_ids = obj_ids[1:]   # First id is the background, so remove it
        masks = mask == obj_ids[:, None, None]  # Split the color encoded mask into a set of binary masks

        num_objs = len(obj_ids)
        boxes = list()
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)   # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self): return len(self.imgs)


class FenceDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'labels'))))

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx:int):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        mask_path = os.path.join(self.root, 'labels', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask = np.array(mask)
        mask = np.where(mask!=0, 1, 0)
        obj_ids = np.unique(mask)   # Instances are encoded as different colors
        obj_ids = obj_ids[1:]   # First id is the background, so remove it
        masks = mask == obj_ids[:, None, None]  # Split the color encoded mask into a set of binary masks

        num_objs = len(obj_ids)
        boxes = list()
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)   # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class LorenzDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = list()
        for f in os.listdir(root):
            if f.endswith('.jpg'):
                self.imgs.append(f)
        self.imgs = sorted(self.imgs)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx:int):
        img_path = os.path.join(self.root, self.imgs[idx])

        img = Image.open(img_path).convert('RGB')

        # mask = np.array(mask)
        # obj_ids = np.unique(mask)   # Instances are encoded as different colors
        # obj_ids = obj_ids[1:]   # First id is the background, so remove it
        # masks = mask == obj_ids[:, None, None]  # Split the color encoded mask into a set of binary masks

        # num_objs = len(obj_ids)
        # boxes = list()
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.ones((num_objs,), dtype=torch.int64) # there is only one class
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)   # suppose all instances are not crowd

        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, {})

        return img, target


def get_instance_segmentation_model(num_classes:int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_transform(train:bool):
    transforms = list()
    transforms.append(T.ToTensor())
    if train: transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def plot_image_w_mask(img, pred)->None:
    fig, ax = plt.subplots(1)
    ax.imshow(transforms.ToPILImage()(img))
    for i, mask in enumerate(pred['masks']):
        mask = mask.mul(255).cpu()
        ax.imshow(transforms.ToPILImage()(mask), alpha=0.5)
        x = pred['boxes'][i][0].item()
        y = pred['boxes'][i][1].item()
        width = pred['boxes'][i][2].item() - pred['boxes'][i][0].item()
        height = pred['boxes'][i][3].item() - pred['boxes'][i][1].item()
        rect = patches.Rectangle((x,y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


def split_image_in_patches(img_path, patch_size:int):
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = img.shape
    rows = h // patch_size
    cols = w // patch_size
    patches = list()
    for i in range(0, rows):
        for j in range(0, cols):
            ymin = i * h // rows
            ymax = i * h // rows + h // rows
            xmin = j * w // cols
            xmax = j * w // cols + w // cols
            patch = img[ymin : ymax, xmin : xmax]
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            patches.append(patch)
    return patches


if __name__ == "__main__":
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')