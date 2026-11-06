# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        topk2 (int): Secondary topk value for additional filtering.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        stride (list): List of stride values for different feature levels.
        stride_val (int): The stride value used for select_candidates_in_gts.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-9,
        topk2=None,
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            stride (list, optional): List of stride values for different feature levels.
            eps (float, optional): A small value to prevent division by zero.
            topk2 (int, optional): Secondary topk value for additional filtering.
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Move tensors to CPU, compute, then move back to original device
                LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
                cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
                result = self._forward(*cpu_tensors)
                return tuple(t.to(device) for t in result)
            raise

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.
            align_metric (torch.Tensor): Alignment metric for selecting best matches.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices  # (b, n_max_boxes)
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos

# 新增修改1：SAQ-DPS：尺度自适应质量门控的训练期 O2M 密集正样本监督
class SAQTaskAlignedAssigner(TaskAlignedAssigner):
    """Scale-Adaptive Quality-Gated Task-Aligned Assigner.

    SAQ-DPS:
        Scale-Adaptive Quality-Gated Dense Positive Supervision

    作用：
        只用于训练期 one-to-many 分支。
        根据 GT 尺度动态调整每个目标的正样本数量，并用中心先验对 align_metric 做质量门控。

    与原 TaskAlignedAssigner 的区别：
        1. 原方法：每个 GT 使用固定 top-k。
        2. SAQ：小目标使用更大的 K，大目标使用较小的 K。
        3. 原方法：align_metric = cls_score^alpha * IoU^beta。
        4. SAQ：align_metric = cls_score^alpha * IoU^beta * center_prior^center_alpha。
    """

    def __init__(
        self,
        topk: int = 12,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-9,
        topk2=None,
        k_min: int = 3,
        tiny_thr: float = 16.0,
        eta: float = 0.5,
        center_radius: float = 2.5,
        center_alpha: float = 1.0,
    ):
        super().__init__(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )

        # topk 是最大正样本数 Kmax
        self.k_min = min(int(k_min), int(topk))
        self.tiny_thr = float(tiny_thr)
        self.eta = float(eta)

        # center_radius 以最小 stride 为基准，得到中心先验的最小半径
        self.center_radius = float(center_radius)
        self.center_alpha = float(center_alpha)

    def get_dynamic_ks(self, gt_bboxes, mask_gt):
        """Generate dynamic K for each GT according to target scale.

        Args:
            gt_bboxes: [B, M, 4], xyxy, image-scale.
            mask_gt: [B, M, 1].

        Returns:
            dynamic_ks: [B, M], long.
        """
        wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp(min=1.0)
        scale = torch.sqrt((wh[..., 0] * wh[..., 1]).clamp(min=1.0))

        # 目标越小，K 越接近 topk；目标越大，K 逐渐减小到 k_min。
        k = self.topk * (self.tiny_thr / (scale + self.eps)).pow(self.eta)
        k = k.round().clamp(min=self.k_min, max=self.topk).long()

        valid = mask_gt.squeeze(-1).bool()
        k = torch.where(valid, k, torch.zeros_like(k))
        return k

    def get_center_prior(self, anc_points, gt_bboxes, mask_gt):
        """Compute center quality prior for candidate anchors.

        Args:
            anc_points: [A, 2], image-scale anchor centers.
            gt_bboxes: [B, M, 4], xyxy, image-scale.
            mask_gt: [B, M, 1].

        Returns:
            center_prior: [B, M, A].
        """
        gt_center = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) * 0.5
        wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp(min=1.0)
        scale = torch.sqrt((wh[..., 0] * wh[..., 1]).clamp(min=1.0))

        min_stride = float(self.stride[0]) if len(self.stride) else 8.0
        min_radius = min_stride * self.center_radius

        # 小目标使用最小中心半径，避免半径过小导致几乎没有候选点。
        radius = torch.clamp(scale * 0.5, min=min_radius).unsqueeze(-1)

        dist2 = ((anc_points.view(1, 1, -1, 2) - gt_center.unsqueeze(2)) ** 2).sum(-1)
        center_prior = torch.exp(-dist2 / (2.0 * radius.pow(2) + self.eps))

        return center_prior * mask_gt.float()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask with scale-adaptive dynamic K and quality-gated metric."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)

        # 原始 TAL metric: cls_score^alpha * IoU^beta
        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            mask_in_gts * mask_gt,
        )

        # 中心质量门控：抑制远离目标中心的低质量密集正样本。
        center_prior = self.get_center_prior(anc_points, gt_bboxes, mask_gt)
        align_metric = align_metric * center_prior.pow(self.center_alpha)

        # 每个 GT 动态 K。
        dynamic_ks = self.get_dynamic_ks(gt_bboxes, mask_gt)

        mask_topk = self.select_topk_candidates(
            align_metric,
            dynamic_ks=dynamic_ks,
        )

        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None, dynamic_ks=None):
        """Select dynamic top-k candidates.

        Args:
            metrics: [B, M, A].
            dynamic_ks: [B, M]. Each GT has its own K.

        Returns:
            count_tensor: [B, M, A].
        """
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)

        if dynamic_ks is not None:
            rank = torch.arange(self.topk, device=metrics.device).view(1, 1, -1)
            valid_topk = rank < dynamic_ks.clamp(max=self.topk).unsqueeze(-1)
            valid_metric = topk_metrics > self.eps
            topk_mask = valid_topk & valid_metric
        elif topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

        topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)

        for k in range(self.topk):
            src = topk_mask[:, :, k : k + 1].to(torch.int8)
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], src)

        # 避免同一 GT 内重复计数。
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        gt_bboxes_clone = gt_bboxes.clone()
        wh_mask = gt_bboxes_clone[..., 2:4] < self.stride[0]
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
            gt_bboxes_clone[..., 2:4],
        )

        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes_clone)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """Transform rotated bounding box (xywh) to distance (ltrb). This is the inverse of dist2rbox.

    Args:
        target_bboxes (torch.Tensor): Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        target_angle (torch.Tensor): Target angle with shape (bs, h*w, 1).
        dim (int, optional): Dimension along which to split.
        reg_max (int, optional): Maximum regression value for clamping.

    Returns:
        (torch.Tensor): Rotated distance with shape (bs, h*w, 4), format [l, t, r, b].
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist
