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
mim test mmcls configs/estimation/resnet18_8xb16_cifar10.py \
    --checkpoint cifar10/latest.pth \
    --work-dir cifar10 --gpus 1
```

### Retrain

```bash
mim gridsearch mmcls configs/retrain/resnet18_8xb16_cifar10.py \
    --work-dir cifar10 --gpus 1 \
    --cfg-options load_from=cifar10/latest.pth \
    --search-args '--model.test_cfg.mask_ratio 0.1 0.3 0.5 0.7 0.9'
```

## SVHN

### Data preparation

You have to convert `SVHN` to `CustomDataset` for retraining.

```bash
python tools/dataset_converters/svhn2custom.py -o data/svhn
```

### Train

```bash
mim train mmcls configs/train/resnet18_8xb16_svhn.py \
    --work-dir svhn --gpus 1
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
    --search-args '--model.test_cfg.mask_ratio 0.1 0.3 0.5 0.7 0.9'
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
mim test mmcls test/resnet50_8xb8_cub.py --checkpoint cub/resnet50_8xb8_cub_20220307-57840e60.pth \
    --work-dir cub --gpus 1
```

### Retrain

```bash
mim gridsearch mmcls cub/resnet50_8xb8_cub.py \
    --work-dir cub --gpus 1 \
    --cfg-options load_from=cub/resnet50_8xb8_cub_20220307-57840e60.pth \
    --search-args '--model.test_cfg.mask_ratio 0.1 0.3 0.5 0.7 0.9'
```
