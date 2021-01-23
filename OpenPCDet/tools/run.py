import os, pathlib
import itertools
from tqdm import tqdm
import numpy as np
np.random.seed(1024)

import torch, numba, datetime
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file

def limit_period(val, offset=0.5, period=np.pi):
  return val - np.floor(val / period + offset) * period


class Runner(object):
	def __init__(self, cfg_file, ckpt=''):

		cfg_from_yaml_file(cfg_file, cfg)
		log_file = ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
		logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

		test_set, test_loader, _ = build_dataloader(
			dataset_cfg=cfg.DATA_CONFIG,
			class_names=cfg.CLASS_NAMES,
			batch_size=1,
			dist=False, workers=8, training=False, logger=logger
		)

		model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
		# load checkpoint
		model.load_params_from_file(filename=ckpt, to_cpu=False, logger=logger)
		model.cuda()
		model.eval()

		self.model = model
		self.dataloader = test_loader

	def forward(self, sample_idx, extra_rois=None):
		index = self.dataloader.dataset.sample_id_list.index(sample_idx)

		# batch_dict = next(itertools.islice(self.dataloader, index, None))
		# assert sample_idx == batch_dict['frame_id'][0]

		tt = self.dataloader.dataset.__getitem__(index)
		batch_dict = self.dataloader.dataset.collate_batch([tt])
		assert sample_idx == batch_dict['frame_id'][0]

		if extra_rois is not None:
			batch_dict['extra_rois'] = extra_rois

		with torch.no_grad():
			load_data_to_gpu(batch_dict)
			pred_dicts, _ = self.model(batch_dict)

		if extra_rois is not None:
			scores = pred_dicts[0]['pred_scores']
			boxes3d = pred_dicts[0]['pred_boxes']
			iscores = batch_dict.pop('batch_cls_preds')
			iboxes3d = batch_dict.pop('batch_box_preds')
			iboxes3d = iboxes3d.squeeze()
			iscores = iscores.squeeze()

			extra_boxes3d = batch_dict.pop('extra_rois')
			extra_scores = batch_dict.pop('extra_scores')

			return iboxes3d, iscores, boxes3d, scores, extra_boxes3d, extra_scores
		else:
			det_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
			det_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
			det_boxes[:, 6] = limit_period(det_boxes[:, 6], 0.5, 2 * np.pi)

			return det_boxes, det_scores


if __name__ == '__main__':
	mot_detector = Runner('../output/kitti_models/pv_rcnn_trk/pvrcnn_car_rtrain/pv_rcnn_trk.yaml',
	                      '../output/kitti_models/pv_rcnn_trk/pvrcnn_car_rtrain/ckpt/checkpoint_epoch_80.pth')
	mot_detector.forward('0001-000010')
