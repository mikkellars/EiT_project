"""
This script splits data into train and test set.
"""


import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Script for splitting data into train and test')
    parser.add_argument('--input_dir', type=str, default='/home/mathias/Documents/scape_data/LUK3-L-02204-0G07-20-Bin2/upper-left', help='path/to/data')
    parser.add_argument('--output_dir', type=str, default='/home/mathias/Documents/scape_data/LUK3-L-02204-0G07-20-Bin2/upper-left', help='path/to/put/txt/files')
    parser.add_argument('--split_train', action='store_true', help='If True, only split into train and val')
    args = parser.parse_args()
    return args


def get_file_list_from_dir(data_dir):
    all_files = os.listdir(os.path.abspath(data_dir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files


def write_to_file(data, file_name):
    f = open(file_name, 'w+')
    for d in data:
        f.write(f'{d}\n')
    f.close()


def split_data(input_dir: str, output_dir: str, split_train: bool = False):
    """Splits data into train, validation, and test (or just train and validation if split_train is True).

    Args:
        input_dir (str): Directory with images.
        output_dir (str): Directory to place txt files.
        split_train (bool, optional): Split only for train, thus train and validation. Defaults to False.
    """

    assert os.path.exists(input_dir), 'the input dir does not exsists'
    assert os.path.exists(output_dir), 'the output dir does not exists'

    data_files = get_file_list_from_dir(input_dir)
    data_files = [s.replace('.png', '') for s in data_files]
    data_files = np.array(data_files)

    data_len = len(data_files)

    if split_train is False:
        train_idxs = np.floor(0.5 * data_len).astype(np.int32)
        test_idxs = data_len - train_idxs

        train_len = train_idxs
        train_idxs = np.floor(0.7 * train_len).astype(np.int32)
        val_idxs = train_len - train_idxs

        idx = np.hstack((np.ones(test_idxs)+1, np.ones(val_idxs), np.zeros(train_idxs))).astype(np.int32)
        np.random.shuffle(idx)

        test_set = data_files[idx==2]
        val_set = data_files[idx==1]
        train_set = data_files[idx==0]
        assert set(train_set) != set(test_set) != set(val_set), 'something went wrong'

        write_to_file(test_set, f'{output_dir}/test.txt')
        write_to_file(train_set, f'{output_dir}/train.txt')
        write_to_file(val_set, f'{output_dir}/val.txt')

    elif split_train is True:
        train_idxs = np.floor(0.7 * data_len).astype(np.int32)
        val_idxs = data_len - train_idxs

        idx = np.hstack((np.ones(val_idxs), np.zeros(train_idxs))).astype(np.int32)
        np.random.shuffle(idx)

        val_set = data_files[idx==1]
        train_set = data_files[idx==0]

        assert set(train_set) != set(val_set), 'something went wrong'

        write_to_file(train_set, f'{output_dir}/train.txt')
        write_to_file(val_set, f'{output_dir}/val.txt')

    print('Done!')


if __name__ == '__main__':
    args = parse_arguments()
    split_data(args.input_dir, args.output_dir, args.split_train)
