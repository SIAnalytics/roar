import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import mmcv

from mmcls.datasets import CIFAR10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CIFAR to CustomDataset format')
    parser.add_argument('-o', '--out-dir', required=True, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tmpdir = TemporaryDirectory()
    cifar = CIFAR10(tmpdir.name, test_mode=False)
    for data in cifar:
        base_dir = osp.join(args.out_dir, 'train', str(data['gt_label']))
        mmcv.imwrite(data['img'],
                     osp.join(base_dir,
                              str(data['sample_idx']) + '.png'))

    cifar = CIFAR10(tmpdir.name, test_mode=True)
    for data in cifar:
        base_dir = osp.join(args.out_dir, 'test', str(data['gt_label']))
        mmcv.imwrite(data['img'],
                     osp.join(base_dir,
                              str(data['sample_idx']) + '.png'))
    tmpdir.cleanup()


if __name__ == '__main__':
    main()
