"""
This script splits data into train and test set.
"""


import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for splitting data into train and test')

    parser.add_argument('--input_dir', type=str, default='data/fence_data/train_set/images', help='path/to/data')
    parser.add_argument('--output_dir', type=str, default='data/fence_data/train_set', help='path/to/put/txt/files')
    parser.add_argument('--ext', type=str, default='jpg', help='file extension')
    parser.add_argument('--split_train', type=float, default=0.3, help='')

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def get_file_list_from_dir(data_dir, ext):
    all_files = os.listdir(os.path.abspath(data_dir))
    data_files = list(filter(lambda file: file.endswith(f'.{ext}'), all_files))
    return data_files


def write_to_file(data, file_name):
    f = open(file_name, 'w+')
    for d in data:
        f.write(f'{d}\n')
    f.close()


def main(args):
    assert os.path.exists(args.input_dir), 'the input dir does not exsists'
    assert os.path.exists(args.output_dir), 'the output dir does not exists'

    data_path = args.input_dir
    data_files = get_file_list_from_dir(data_path, args.ext)
    data_files = [s.replace(f'.{args.ext}', '') for s in data_files]
    data_files = np.array(data_files)

    data_len = len(data_files)
    train_idxs = np.floor(args.split_train * data_len).astype(np.int32)
    test_idxs = np.ceil((1 - args.split_train) * data_len).astype(np.int32)
    
    idx = np.hstack((np.ones(train_idxs), np.zeros(test_idxs))).astype(np.int32)
    np.random.shuffle(idx)

    train_set = data_files[idx==1]
    test_set = data_files[idx==0]

    assert set(train_set) != set(test_set), 'something went wrong'

    write_to_file(train_set, f'{args.output_dir}/val.txt')
    write_to_file(test_set, f'{args.output_dir}/train.txt')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
