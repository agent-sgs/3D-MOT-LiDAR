import numpy as np, math
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from mmdet.ops.iou3d import iou3d_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_sampler_cfg = edict(
            ROI_PER_IMAGE=128,
            FG_RATIO=0.5,
            CLS_FG_THRESH=0.6,
            CLS_BG_THRESH=0.45,
            CLS_BG_THRESH_LO=0.1,
            HARD_BG_RATIO=0.8,
        )
        self.aug_noise = edict(
            rot_range=-np.pi / 4,
            loc_range=1.5,
        )
        self.bev_range = [0, -40., 70.4, 40.,]

    def forward(self, rois, gt_boxes):
        batch_size = len(rois)
        batch_rois = []
        batch_labels = []
        for index in range(batch_size):
            cur_roi, cur_gt = rois[index], gt_boxes[index]
            assert len(cur_gt) > 0

            aug_rois = torch.cat([cur_roi.clone().detach(), cur_gt.clone().detach()], dim=0)
            aug_rois = self.aug_roi_by_noise(aug_rois)

            aug_rois, aug_overs, aug_gt_of_roi = self.sample_rois_for_rcnn(aug_rois, cur_gt)

            aug_labels = torch.zeros_like(aug_overs, dtype=torch.int32)
            aug_labels[aug_overs > self.roi_sampler_cfg.CLS_FG_THRESH] = 1
            ignore_mask = (aug_overs > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (aug_overs < self.roi_sampler_cfg.CLS_FG_THRESH)
            aug_overs[ignore_mask] = -1

            batch_rois.append(aug_rois)
            batch_labels.append(aug_labels)

            # from SA_SSD.tools.viz_utils import draw_lidar, draw_boxes3d
            # from SA_SSD.mmdet.core.bbox3d.geometry import center_to_corner_box3d
            # gt_corners = center_to_corner_box3d(cur_gt.cpu().numpy())
            # roi_corners = center_to_corner_box3d(cur_roi.detach().cpu().numpy())
            # aug_corners = center_to_corner_box3d(aug_rois.cpu().numpy())
            #
            # fig = draw_boxes3d(aug_corners, color=(1., 0., 0.))
            # fig = draw_boxes3d(gt_corners, fig, color=(0., 1., 0.))
            # fig = draw_boxes3d(roi_corners, fig, color=(0., 0., 1.))

        return batch_rois, batch_labels

    def sample_rois_for_rcnn(self, cur_roi, cur_gt):
        iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
        max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

        sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

        return cur_roi[sampled_inds], max_overlaps[sampled_inds], cur_gt[gt_assignment[sampled_inds]]

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        # fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)
        fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise(self, rois):

        times = math.ceil(self.roi_sampler_cfg.ROI_PER_IMAGE * 2 / len(rois))
        aug_rois = rois.repeat(times, 1)

        rot_noises = (torch.rand(len(aug_rois)) - 0.5) / 0.5 * self.aug_noise.rot_range
        loc_noises = (torch.rand(len(aug_rois), 2) - 0.5) / 0.5 * self.aug_noise.loc_range

        aug_rois[:, :2] += loc_noises.to(rois)
        aug_rois[:, 6] += rot_noises.to(rois)

        mask = (aug_rois[:, 0] > self.bev_range[0]) & (aug_rois[:, 0] < self.bev_range[2]) & \
                    (aug_rois[:, 1] > self.bev_range[1]) & (aug_rois[:, 1] < self.bev_range[3])
        aug_rois = aug_rois[mask]

        aug_rois = torch.cat([aug_rois, rois], dim=0)
        return aug_rois





