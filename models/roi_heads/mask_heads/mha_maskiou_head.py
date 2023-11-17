# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, Linear, MaxPool2d
from mmcv.runner import BaseModule, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer


@HEADS.register_module()
class SparseMaskIoUHead(BaseModule):
    """Mask IoU Head.
    This head predicts the IoU of predicted masks and corresponding gt masks.
    """

    def __init__(self,
                 num_convs=3,
                 num_fcs=2,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=80,
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256, # 256 + 1
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=14,
                     with_proj=False,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_iou=dict(type='MSELoss', loss_weight=0.5),
                 init_cfg=[
                     dict(type='Kaiming', override=dict(name='convs')),
                     dict(type='Caffe2Xavier', override=dict(name='fcs')),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='fc_mask_iou'))
                 ]):
        super(SparseMaskIoUHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.fp16_enabled = False

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature and mask prediction
                # in_channels = self.in_channels + 1
                in_channels = self.in_channels
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=stride,
                    padding=1))

        roi_feat_size = _pair(roi_feat_size)
        pooled_area = (roi_feat_size[0] // 2) * (roi_feat_size[1] // 2)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (
                self.conv_out_channels *
                pooled_area if i == 0 else self.fc_out_channels)
            self.fcs.append(Linear(in_channels, self.fc_out_channels))

        self.fc_mask_iou = Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = MaxPool2d(2, 2)
        self.loss_iou = build_loss(loss_iou)

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.reduce_conv = Conv2d(self.in_channels + 1, self.in_channels, 3, stride=1, padding=1)

    def forward(self, mask_feat, mask_pred, attn_feat):
        mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))

        mask_feat = torch.cat((mask_feat, mask_pred_pooled), 1)
        mask_feat = self.reduce_conv(mask_feat)
        mask_feat = self.relu(mask_feat)

        attn_feat = attn_feat.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            attn_feat, mask_feat)

        x = proposal_feat_iic.permute(0, 2, 1).reshape(mask_feat.size())
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        mask_iou = torch.sigmoid(mask_iou)
        return mask_iou

    @force_fp32(apply_to=('mask_iou_pred', ))
    def loss(self, mask_iou_pred, mask_iou_targets):
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(mask_iou_pred[pos_inds],
                                          mask_iou_targets[pos_inds])
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)

    @force_fp32(apply_to=('mask_pred', ))
    def get_targets(self, sampling_results, gt_masks, mask_pred, mask_targets,
                    rcnn_train_cfg):
        """Compute target of mask IoU.
        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.
        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (BitmapMask | PolygonMask): Gt masks (the whole instance)
                of each image, with the same shape of the input image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.
        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        # compute the area ratio of gt areas inside the proposals and
        # the whole instance
        area_ratios = map(self._get_area_ratio, pos_proposals,
                          pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_pred = (mask_pred > rcnn_train_cfg.mask_thr_binary).float()
        mask_pred_areas = mask_pred.sum((-1, -2))

        # mask_pred and mask_targets are binary maps
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))

        # compute the mask area of the whole instance
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)

        mask_iou_targets = overlap_areas / (
            mask_pred_areas + gt_full_areas - overlap_areas)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance."""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # compute mask areas of gt instances (batch processing for speedup)
            gt_instance_mask_area = gt_masks.areas
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                # crop the gt mask inside the proposal
                bbox = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask.crop(bbox) # 这里有问题gt_mask_in_proposal可能为空
                if len(gt_mask_in_proposal.areas)==0:
                    ratio = 1e-7
                else:
                    ratio = gt_mask_in_proposal.areas[0] / (
                        gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
                pos_proposals.device)
        else:
            area_ratios = pos_proposals.new_zeros((0, ))
        return area_ratios

    @force_fp32(apply_to=('mask_iou_pred', ))
    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.
        mask_score = bbox_score * mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = mask_iou_pred[inds, det_labels] * det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [mask_scores[det_labels == i] for i in range(self.num_classes)]