_base_ = '../train/resnet18_8xb16_svhn.py'

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='RemOve',
        mask_dir='svhn/train',
        attr='',  # placeholder for gridsearch
        ratio=0,  # placeholder for gridsearch
        filter='',  # placeholder for gridsearch
        mean=[125.307, 122.961, 113.8575]),
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='RemOve', mask_dir='svhn/test', mean=[125.307, 122.961,
                                                   113.8575]),
    dict(type='PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
