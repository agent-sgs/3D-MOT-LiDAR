import sys
sys.path.append('.')
sys.path.append('..')

import os, pathlib
import cv2
import argparse, time
from tqdm import tqdm
import pickle, shutil
from lib.tracker3d_oxts import Tracker3D
from lib.common_utils import oxts_kitti
from lib.common_utils.calibration_kitti import Calibration
from lib.common_utils.box_utils import kitti_bbox2results
from lib.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_cpu
import torch
from OpenPCDet.tools.run import Runner

DETECTOR = 'PointRCNN'
assert DETECTOR in ['PVRCNN', 'PointRCNN']

data_path = pathlib.Path("/opt/projects/data/kitti/track/")

sequence_train_ids = ['0000', '0002', '0003', '0004', '0005', '0007', '0009', '0011', '0017', '0020']
sequence_val_ids = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']
sequence_test_ids = ["{:04d}".format(s) for s in range(29)]


def parse_args():
  parser = argparse.ArgumentParser(description='3D MOT System')
  parser.add_argument('tag', type=str, default='car_val', help='MOT tag for tuning')
  parser.add_argument('split', type=str, default='val', help='selecting from train val test')

  parser.add_argument('--min_hits', type=int, default=3, help='trajectory creation param')
  parser.add_argument('--max_ages', type=int, default=2, help='trajectory deletion param')
  parser.add_argument('--acc_iou_thr', type=float, default=0.01, help='association param')
  parser.add_argument('--high_score', type=float, default=0.9, help='trajectory creation param')
  parser.add_argument('--low_score', type=float, default=0.1, help='trajectory output param')

  args = parser.parse_args()
  return args
args = parse_args()
assert args.split in ['train', 'val', 'test']
if args.split == 'val':
  eval_sequences = sequence_val_ids
  oxts_dir = data_path / 'training' / 'oxts'
  calib_dir = data_path / 'training' / 'calib'
  image_dir = data_path / 'training' / 'image_02'
elif args.split == 'train':
  eval_sequences = sequence_train_ids
  oxts_dir = data_path / 'training' / 'oxts'
  calib_dir = data_path / 'training' / 'calib'
  image_dir = data_path / 'training' / 'image_02'
elif args.split == 'test':
  eval_sequences = sequence_test_ids
  oxts_dir = data_path / 'testing' / 'oxts'
  calib_dir = data_path / 'training' / 'calib'
  image_dir = data_path / 'testing' / 'image_02'

with open(str(data_path / 'training' / 'oxts_0001_177_180.pkl'), 'rb') as f:
  oxts_0001_177_180 = pickle.load(f)

# eval_dir = pathlib.Path('/home/sgs/rtmot/oxtsA') / DETECTOR / ('Car_'+args.split+'_h'+str(args.high_score)) / 'data'
# eval_dir = pathlib.Path('/home/sgs/rtmot/oxtsA') / DETECTOR / ('Car_'+args.split+'_h'+str(args.high_score)+'_rinit') / 'data'
eval_dir = pathlib.Path('/home/sgs/rtmot/FoxtsA') / DETECTOR / ('Car_'+args.split+'_h'+str(args.high_score)+'_tc1.1') / 'data'
if os.path.exists(eval_dir): shutil.rmtree(eval_dir)
os.makedirs(eval_dir)
print('#'*50+'\n', str(eval_dir))

###  3D DET  ###
if DETECTOR == 'PVRCNN':
  ckpt_file = '/opt/projects/RTMOT/OpenPCDet/output/kitti_models/pv_rcnn_trk/pvrcnn_car_rtrain/ckpt/checkpoint_epoch_80.pth'
  mot_detector = Runner('/opt/projects/RTMOT/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn_trk.yaml', ckpt_file)
elif DETECTOR == 'PointRCNN':
  ckpt_file = '/opt/projects/RTMOT/OpenPCDet/output/kitti_models/pointrcnn_iou_trk/pointrcnniou_car_rtrain/ckpt/checkpoint_epoch_80.pth'
  mot_detector = Runner('/opt/projects/RTMOT/OpenPCDet/tools/cfgs/kitti_models/pointrcnn_iou_trk.yaml', ckpt_file)
else:
  raise NotImplementedError

###  3D MOT  ###
mot_tracker = Tracker3D(min_hits=args.min_hits, max_ages=args.max_ages,
                        low_score=args.low_score, high_score=args.high_score,
                        acc_iou_thr=args.acc_iou_thr, boxes_iou3d_cpu=boxes_iou3d_cpu)


sum_times, sum_frames = 0, 0
for sdx in eval_sequences:
  eval_file = open(eval_dir / (sdx + '.txt'), 'w')

  calib = Calibration(calib_dir / (sdx+'.txt'), track=True)
  pose_generator = oxts_kitti.PoseGenerator(oxts_dir / (sdx+'.txt'))

  image_ids = [img.strip('.png') for img in os.listdir(image_dir / sdx)]
  image_ids.sort()

  for idx in tqdm(image_ids, desc=sdx):
    image_idx = sdx + '-' + idx

    image = cv2.imread(str(image_dir / sdx / (idx+'.png')))
    meta = {'calib': calib,
            'img_shape': image.shape[:2],
            'oxts': pose_generator.get(int(idx), extend_matrix=True)}

    #--- special case for LiDAR-based 3D MOT system, not for image-related systems:
    if sdx == '0001' and int(idx) in [177, 178, 179, 180]:
      tic = time.time()
      meta['oxts'] = oxts_0001_177_180[int(idx)]
      mot_tracker.special_kf_update()
      trk_ids, trk_bboxes, trk_scores = mot_tracker.special_output(meta)
      toc = time.time()
      sum_times += (toc - tic)
      sum_frames += 1
      trk_labels = ['Car'] * len(trk_bboxes)
      kitti_bbox2results(int(idx), trk_bboxes, trk_scores, trk_labels, trk_ids, meta, eval_file)
      continue

    det_boxes, det_scores = mot_detector.forward(image_idx)
    torch.cuda.synchronize()

    tic = time.time()
    trk_ids, trk_bboxes, trk_scores = mot_tracker.update(det_boxes, det_scores, meta)
    toc = time.time()
    sum_times += (toc - tic)
    sum_frames += 1

    trk_labels = ['Car'] * len(trk_bboxes)
    kitti_bbox2results(int(idx), trk_bboxes, trk_scores, trk_labels, trk_ids, meta, eval_file)
  eval_file.close()
  print(f"  Total frames {sum_frames}, avg_time: {sum_times / sum_frames}, FPS: {sum_frames / sum_times}")
print(f"evaluation results have saved into {eval_dir}")


