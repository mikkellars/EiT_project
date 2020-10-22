"""
"""


import torch
import torchvision
import numpy as np
import fastai.vision as faiv
from PIL import Image
from model import *
from dataset import *


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Training of DeepLabV3+')
    parser.add_argument('--exp', type=str, default='texel', help='name of the experiment')
    parser.add_argument('--data_dir', type=str, default='../data/fence_data/patch_train_set', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='images', help='path to save directory')
    parser.add_argument('--models_dir', type=str, default='', help='path to model directory')
    parser.add_argument('--bs', type=int, default=4, help='batch size (default: 1)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers (default: 8')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--resume_model', type=str, default='texel_model', help='path to resume model')
    parser.add_argument('--decay_margin', type=float, default=0.01, help='margin for starting decay (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.01, help='learning decay of RMSprop (default: 0.01)')
    parser.add_argument('--w_decay', type=float, default=1e-4, help='weight decay RMSprop (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of RMSprop (default: 0.9)')
    args = parser.parse_args()
    return args


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_loss = float('inf')
    best_acc = 0.0

    model = deeplabv3_resnet('resnet18')
    model.to(device)

    dataset = TexelDataset(args.data_dir, 'train', True)
    val_dataset = TexelDataset(args.data_dir, 'val', False)
    data = faiv.DataBunch.create(dataset, val_dataset, bs=args.bs, num_workers=args.workers, worker_init_fn=lambda *_: np.random.seed())

    # learner = faiv.learner.Learner(data, model, loss_func=torch.nn.MSELoss()).to_fp16()
    # if args.resume_model != '': learner.load(args.resume_model)
    # learner.fit_one_cycle(args.epochs, args.lr, wd=args.w_decay)
    # learner.save(f'{args.exp}_model')

    model = deeplabv3_resnet('resnet18')
    model.load_state_dict(torch.load(f'models/{args.exp}_model.pth')['model'])
    model.to(device)

    img_path = '/home/mathias/Documents/experts_in_teams_proj/vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_16.jpg'
    img = Image.open(img_path).convert('RGB')

    img = torchvision.transforms.Resize((512, 512))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(*imagenet_stats)(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img).squeeze(0)
    pred = pred.detach().cpu().numpy()

    pred = Image.fromarray(pred)
    pred.save(f'{args.save_dir}/test.png')


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    main(args)
