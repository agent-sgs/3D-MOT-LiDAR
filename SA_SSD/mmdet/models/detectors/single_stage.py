import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, rbbox2roi, bbox2result, multi_apply, kitti_bbox2results,\
                        tensor2points, delta2rbbox3d, weighted_binary_cross_entropy)
import torch.nn.functional as F
from .proposal_target_layer import ProposalTargetLayer


class SingleStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

        if extra_head is not None:
            self.augment_rois_layer = ProposalTargetLayer()
            self.extra_head = builder.build_single_stage_head(extra_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in [
                'voxels', 'num_points',
            ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes',
            ]:
                ret[key] = elems
            else:
                ret[key] = torch.stack(elems, dim=0)
        return ret

    def forward_train(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6), point_misc = self.neck(vx, ret['coordinates'], batch_size)

        losses = dict()

        aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
        losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], ret['gt_bboxes'], thr=0.1)
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            guided_anchors, labels = self.augment_rois_layer(guided_anchors, ret['gt_bboxes'])
            bbox_score = self.extra_head(conv6, guided_anchors)
            refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
            refine_losses = self.extra_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], None, thr=.1)

        bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True)

        det_bboxes, det_scores = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, img_meta, self.test_cfg.extra)

        results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]

        return results

    def forward_test_org(self, img, img_meta, extra_rois, **kwargs):

        batch_size = len(img_meta)
        assert batch_size == 1

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], None, thr=.1)
        num_rois = guided_anchors[0].shape[0]

        if extra_rois is not None:
            guided_anchors[0] = torch.cat((guided_anchors[0], extra_rois), dim=0)

        bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True)
        bbox_score = [torch.sigmoid(score) for score in bbox_score]

        det_bboxes, det_scores = self.extra_head.get_rescore_bboxes1(
            guided_anchors[0][:num_rois].contiguous(), bbox_score[0][:num_rois].contiguous(), img_meta, self.test_cfg.extra)

        # results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]
        if extra_rois is not None:
            results = {
                'iscores': bbox_score[0][:num_rois].view(-1),
                'iboxes3d': guided_anchors[0][:num_rois],
                'scores': det_scores,
                'boxes3d': det_bboxes,
                'extra_scores': bbox_score[0][num_rois:].view(-1)
            }
        else:
            results = {
                'scores': det_scores,
                'boxes3d': det_bboxes,
            }

        return results

    def forward_test1(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], None, thr=.1)

        bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True)

        det_bboxes, det_scores = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, img_meta, self.test_cfg.extra)

        results = {
            'conv6': conv6,
            'iscores': bbox_score,
            'ibox3d_lidar': guided_anchors,
            'scores': det_scores,
            'box3d_lidar': det_bboxes,
        }

        return results

    def forward_test2(self, batch_dict):

        extra_score, extra_bboxes = self.extra_head(batch_dict['conv6'], batch_dict['extra_rois'], is_test=True)

        for i, scores in enumerate(extra_score):
            if scores.numel > 0:
                extra_score[i] = torch.sigmoid(scores).view(-1)

        return extra_bboxes, extra_score








