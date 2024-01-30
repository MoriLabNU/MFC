# dataset settings
dataset_type = 'CholecSegDataset'
data_root = '../DATA/cholec/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
sim_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 256)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cheloc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 256)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(512, 256)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 256),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SimDataset',
            data_root='../DATA/simulated/',
            img_dir='images',
            ann_dir='labels',
            pipeline=sim_train_pipeline),
        target=dict(
            type='CholecSegDataset',
            data_root='../DATA/cholec/',
            img_dir='img/train',
            ann_dir=None,
            pipeline=cheloc_train_pipeline)),
    val=dict(
        type='CholecSegDataset',
        data_root='../DATA/cholec/',
        img_dir='img/test',
        ann_dir='gt/test',
        pipeline=test_pipeline),
    test=dict(
        type='CholecSegDataset',
        data_root='../DATA/cholec/',
        img_dir='img/test',
        ann_dir='gt/test',
        pipeline=test_pipeline))
