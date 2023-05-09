custom_imports = dict(
    imports=['roar.engine', 'roar.estimation'], allow_failed_imports=False)
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
    dict(type='PackInputs')
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
estimator = [
    dict(type='Grad'),
    dict(type='GI'),
    dict(type='IG'),
    dict(type='SG'),
    dict(type='VG'),
    dict(type='GC', module='backbone.layer4'),
    dict(type='Sobl'),
    dict(type='Rand'),
]
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='FeatureEstimationHook', estimator=estimator))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
log_level = 'INFO'
load_from = None
