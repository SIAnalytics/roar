import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import mmcv

from roar.datasets import SVHN


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SVHN to CustomDataset format')
    parser.add_argument('-o', '--out-dir', required=True, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tmpdir = TemporaryDirectory()
    svhn = SVHN(tmpdir.name, test_mode=False)
    for data in svhn:
        base_dir = osp.join(args.out_dir, 'train', str(data['gt_label']))
        mmcv.imwrite(data['img'],
                     osp.join(base_dir, data['sample_idx'] + '.png'))

    svhn = SVHN(tmpdir.name, test_mode=True)
    for data in svhn:
        base_dir = osp.join(args.out_dir, 'test', str(data['gt_label']))
        mmcv.imwrite(data['img'],
                     osp.join(base_dir, data['sample_idx'] + '.png'))
    tmpdir.cleanup()


if __name__ == '__main__':
    main()
