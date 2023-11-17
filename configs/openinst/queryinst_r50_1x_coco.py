dataset_file = './coco_to_uvo_ins.py'
init_cfg = dict(type='Pretrained', checkpoint='resnet50-0676ba61.pth')
init_cfg=None
tb_hook = dict(type='TensorboardLoggerHook')
_base_ = [
    dataset_file,
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
num_stages = 6
num_proposals = 100
model = dict(
    type='QueryInst',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=init_cfg,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        with_cp=True),
    neck=dict(
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=6,
        num_repeats=6,
        norm='SyncBN'),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseScoreRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64, 128]),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64, 128]),
        bbox_head=[
            dict(
                type='DIIScoreHead',
                num_classes=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(type='L1Loss', loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        mask_head=[
            dict(
                type='DynamicMaskHead',
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                num_convs=4,
                num_classes=1,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=256,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),
                loss_mask=dict(
                    type='DiceLoss',
                    loss_weight=1.0,
                    use_sigmoid=True,
                    activate=False,
                    eps=1e-5)) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1,
                mask_size=28,
            ) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None, rcnn=dict(max_per_img=num_proposals, mask_thr_binary=0.5)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0),
                     'init_proposal_features':dict(lr_mult=1.0, decay_mult=0.0),
                     'init_proposal_bboxes': dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0)
    )

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[8, 11], warmup_iters=1000)
runner = dict(type='EpochBasedRunner', max_epochs=12)

# log setting
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        tb_hook,
    ])

checkpoint_config = dict(interval=3)
resume_from = None
custom_hooks = []
custom_hooks = [dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)]