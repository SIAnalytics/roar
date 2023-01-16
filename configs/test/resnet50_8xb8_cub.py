custom_imports = dict(
    imports=['roar.engine', 'roar.estimation'], allow_failed_imports=False)
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNet', depth=50),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=200,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss')))
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='PackClsInputs')
]
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='CUB',
        data_root='data/CUB_200_2011',
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
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='FeatureEstimationHook', estimator=estimator))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ClsVisualizer', vis_backends=vis_backends)
log_level = 'INFO'
load_from = None
