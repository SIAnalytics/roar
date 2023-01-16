# **RemOve-And-Retrain** is Improper: Data Processing Inequality Perspective

## Installation

You have to install MMClassification with MIM.

```bash
pip install mim
mim install mmcls==1.0.0rc5
```

## CIFAR 10

### Data preparation

You have to convert `CIFAR10` to `CustomDataset` for retraining.

```bash
python tools/dataset_converters/cifar2custom.py -o data/cifar10
```

### Train

```bash
mim train mmcls configs/train/resnet18_8xb16_cifar10.py \
    --work-dir cifar10 --gpus 1
```

### Estimate a feature importance

```bash
# for train dataset
mim test mmcls configs/test/resnet18_8xb16_cifar10.py \
    --checkpoint cifar10/latest.pth \
    --work-dir cifar10/train --gpus 1 --cfg-options test_dataloader.dataset.test_mode=False
# for test dataset
mim test mmcls configs/test/resnet18_8xb16_cifar10.py \
    --checkpoint cifar10/latest.pth \
    --work-dir cifar10/test --gpus 1
```

### Retrain

```bash
mim gridsearch mmcls configs/retrain/resnet18_8xb16_cifar10.py \
    --work-dir cifar10 --gpus 1 \
    --cfg-options load_from=cifar10/latest.pth \
    --search-args '--train_dataloader.dataset.pipeline.1.attr grad gi ig sg vg gc sobl rand \
        --train_dataloader.dataset.pipeline.1.ratio 10 30 50 70 90 \
        --train_dataloader.dataset.pipeline.1.filter none maximum gaussian'
```

## SVHN

### Data preparation

You have to convert `SVHN` to `CustomDataset` for retraining.

```bash
python tools/dataset_converters/svhn2custom.py -o data/svhn
```

### Train

```bash
# for train dataset
mim test mmcls configs/test/resnet18_8xb16_svhn.py \
    --checkpoint svhn/latest.pth \
    --work-dir svhn/train --gpus 1 --cfg-options test_dataloader.dataset.test_mode=False
# for test dataset
mim test mmcls configs/test/resnet18_8xb16_svhn.py \
    --checkpoint svhn/latest.pth \
    --work-dir svhn/test --gpus 1
```

### Estimate a feature importance

```bash
mim test mmcls configs/estimation/resnet18_8xb16_svhn.py \
    --checkpoint svhn/latest.pth \
    --work-dir svhn --gpus 1
```

### Retrain

```bash
mim gridsearch mmcls configs/retrain/resnet18_8xb16_svhn.py \
    --work-dir svhn --gpus 1 \
    --cfg-options load_from=svhn/latest.pth \
    --search-args '--train_dataloader.dataset.pipeline.1.attr grad gi ig sg vg gc sobl rand \
        --train_dataloader.dataset.pipeline.1.ratio 10 30 50 70 90 \
        --train_dataloader.dataset.pipeline.1.filter none maximum gaussian'
```

## CUB-200

### Data preparation

```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz
mkdir data
ln -s CUB_200_2011 data
```

### Train (optional)

```bash
mim download mmcls --config resnet50_8xb8_cub --dest cub
mim train mmcls cub/resnet50_8xb8_cub.py \
    --work-dir cub --gpus 1
```

### Estimate a feature importance

```bash
mim download mmcls --config resnet50_8xb8_cub --dest cub
mim test mmcls test/resnet50_8xb8_cub.py \
    --checkpoint cub/resnet50_8xb8_cub_20220307-57840e60.pth \
    --work-dir cub --gpus 1
```

### Retrain

```bash
mim gridsearch mmcls configs/retrain/resnet50_8xb8_cub.py \
    --work-dir cub --gpus 1 \
    --cfg-options load_from=cub/resnet50_8xb8_cub_20220307-57840e60.pth \
    --search-args '--train_dataloader.dataset.pipeline.1.attr grad gi ig sg vg gc sobl rand \
        --train_dataloader.dataset.pipeline.1.ratio 10 30 50 70 90 \
        --train_dataloader.dataset.pipeline.1.filter none maximum gaussian'
```
