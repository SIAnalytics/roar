custom_imports = dict(
    imports=['roar.datasets.transforms'], allow_failed_imports=False)
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNet_CIFAR', depth=18),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
data_preprocessor = dict(
    num_classes=10,
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='RemOve',
        mask_dir='svhn/test',
        attr='',  # placeholder
        ratio=0,  # placeholder
        filter='',  # placeholder
        mean=[125.307, 122.961, 113.8575],
        apply_mask=False),
    dict(type='LinearImputation'),
    dict(type='PackClsInputs')
]
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        data_root='data/svhn/test',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, ))
test_cfg = dict()
default_scope = 'mmcls'
default_hooks = dict(timer=dict(type='IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ClsVisualizer', vis_backends=vis_backends)
log_level = 'INFO'
load_from = None
