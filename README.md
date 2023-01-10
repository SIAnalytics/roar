# **RemOve-And-Retrain** is Improper: Data Processing Inequality Perspective

## Installation

You have to install MMClassification with MIM.

```bash
pip install mim
mim install mmcls==1.0.0rc5
```

## CIFAR 10

## SVHN

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
python tools/estimate.py
```

### Retrain

```
mim gridsearch mmcls cub/resnet50_8xb8_cub.py \
    --work-dir cub --gpus 1 \
    --cfg-options load_from=cub/resnet50_8xb8_cub_20220307-57840e60.pth \
    --search-args '--model.test_cfg.mask_ratio 0.1 0.3 0.5 0.7 0.9'
```
