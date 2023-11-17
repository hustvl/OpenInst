# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoSplitDataset',
        is_class_agnostic=True, 
        train_class='all',
        eval_class='all',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type='UVODataset',
        is_class_agnostic=True, 
        train_class='all',
        eval_class='all',
        ann_file='data/UVO/ann/UVO_frame_val.json',
        img_prefix='data/UVO/images/',
        pipeline=test_pipeline),
    test=dict(
        type='UVODataset',
        is_class_agnostic=True, 
        train_class='all',
        eval_class='all',
        ann_file='data/UVO/ann/UVO_frame_val.json',
        img_prefix='data/UVO/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
