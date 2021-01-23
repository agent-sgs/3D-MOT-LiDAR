import os, sys
sys.path.append('..')


import torch, torch.utils.data
import mmcv, pickle, itertools
from .train_utils import load_params_from_file
from mmcv.parallel import MMDataParallel, collate
from mmdet.datasets import utils
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector
from mmdet.ops.iou3d.iou3d_utils import boxes_iou_bev, boxes_iou3d_gpu, boxes_overlap_bev

import numpy as np
def limit_period(val, offset=0.5, period=np.pi):
  return val - np.floor(val / period + offset) * period


class Runner(object):
  def __init__(self, cfg_file, checkpoint='../saved_model_vehicle/checkpoint_epoch_80.pth'):
    cfg = mmcv.Config.fromfile(cfg_file)
    cfg.model.pretrained = None

    # build dataloader
    dataset = utils.get_dataset(cfg.data.val)
    self.class_names = cfg.data.val.class_names
    print("To initize generate detections for ", self.class_names)
    self.dataloader = build_dataloader(dataset, 1, 6, num_gpus=1, shuffle=False, dist=False)

    # build model
    self.model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    self.model = MMDataParallel(self.model, device_ids=[0])
    load_params_from_file(self.model, checkpoint)
    self.model.eval()

  def forward(self, sample_idx, extra_rois=None):
    index = self.dataloader.dataset.sample_ids.index(sample_idx)

    # data = next(itertools.islice(self.dataloader, index, None))
    # assert sample_idx == data['img_meta'].data[0][0]['sample_idx']

    data = collate([self.dataloader.dataset.__getitem__(index)])
    assert sample_idx == data['img_meta'].data[0][0]['sample_idx']

    if extra_rois is not None:
      extra_rois = torch.from_numpy(extra_rois).float().cuda()
      torch.cuda.synchronize()
    with torch.no_grad():
      batch_dict = self.model(return_loss=False, mot=True, extra_rois=extra_rois, **data)

    if extra_rois is not None:
      iboxes3d = batch_dict['iboxes3d']
      iscores = batch_dict['iscores']
      det_scores = batch_dict['scores']
      det_boxes3d = batch_dict['boxes3d']
      extra_scores = batch_dict['extra_scores']
      return iboxes3d, iscores, det_boxes3d, det_scores, extra_rois, extra_scores
    else:
      det_boxes = batch_dict['boxes3d'].detach().cpu().numpy()
      det_scores = batch_dict['scores'].detach().cpu().numpy()
      det_boxes[:, 6] = limit_period(det_boxes[:, 6], 0.5, 2 * np.pi)
      return det_boxes, det_scores

