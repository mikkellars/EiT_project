"""
Training of ResNet50 for texel dataset.
"""

from fastai.vision.all import *

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='model', help='name of trained model')
    parser.add_argument('--data_root', type=str,
                        default='/home/mikkel/Documents/data/scape_data/LUK3-L-02204-0G07-20-Bin2/glico_gen/baseline_crop',
                        help='path to dataset either downloaded or to be downloaded')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 8)')
    parser.add_argument('--save_dir_model', type=str, default='baseline/resnet50/models', help='path to save models')
    parser.add_argument('--save_dir_logs', type=str, default='baseline/resnet50/logs', help='logs path for tensorboard')
    parser.add_argument('--load_model_param', type=str, default='baseline/resnet50/hparams.pt')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 100)')
    args = parser.parse_args()
    return args


def main(args):
    start_time = time.time()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET_PATH = Path("/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/texel_data")

    mask_datablock = DataBlock(
        get_items=get_image_files,
        get_y=parent_label,
        blocks=(ImageBlock, CategoryBlock),
        item_tfms=RandomResizedCrop(224, min_scale=0.3),
        splitter=RandomSplitter(valid_pct=0.2, seed=100),
        batch_tfms=aug_transforms(mult=2)
    )

    dls = mask_datablock.dataloaders(DATASET_PATH)

    learn = cnn_learner(dls, resnet50, metrics=error_rate)
    learn.fine_tune(4)


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    main(args)
