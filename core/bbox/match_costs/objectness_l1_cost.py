import torch
import torch.nn.functional as F

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class ObjectnessL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = L1Cost()
         >>> bbox_pred = torch.rand(10, 1)
         >>> gt_bboxes= torch.FloatTensor([0.8, 0.9])
         >>> self(bbox_pred, gt_bboxes)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., iou_mode='iou'):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, cls_pred, bboxes, gt_bboxes):
        """
        Args:
            cls_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_targets (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # overlaps: [num_gt, num_bboxes]
        overlaps = bbox_overlaps(
            gt_bboxes, bboxes, mode=self.iou_mode, is_aligned=False)
        gt_targets, _ = torch.max(overlaps, 1, keepdim=True) # [num_gt, 1]

        objectness_cost = torch.cdist(cls_pred, gt_targets, p=1)
        return objectness_cost * self.weight