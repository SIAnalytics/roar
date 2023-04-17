_base_ = 'mmpretrain::resnet/resnet50_8xb8_cub.py'

custom_imports = dict(
    imports=['roar.datasets.transforms'], allow_failed_imports=False)
default_scope = 'mmpretrain'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RemOve',
        mask_dir='cub',
        attr='',  # placeholder for gridsearch
        ratio=0,  # placeholder for gridsearch
        filter='',  # placeholder for gridsearch
        mean=[103.53, 116.28, 123.675]),
    dict(type='Resize', scale=600),
    dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RemOve', mask_dir='cub', mean=[103.53, 116.28, 123.675]),
    dict(type='Resize', scale=600),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackClsInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
