import copy
import os.path as osp
from tqdm import tqdm
import numpy as np
import argparse
from kitti_object_eval_python import eval as kitti_eval

sequence_val_ids = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']
sequence_trainval_ids = ["{:04d}".format(s) for s in range(21)]


def parse_args():
  parser = argparse.ArgumentParser(description='3D Object Evaluateion Tool for MOT Dataset')
  parser.add_argument('split', type=str, default='val', help='selecting from train val test')
  parser.add_argument('result_path', type=str, help='')
  parser.add_argument('--framewise', action='store_true', default=False, help='specify the result format: trajectories or framewise ?')
  parser.add_argument('--label_path', type=str, default='../data/training/label_02', help='training label file path')
  parser.add_argument('--class_name', type=str, default='Car', help='classes to be evaluated')
  args = parser.parse_args()
  return args

args = parse_args()


def evaluation(gt_label_dir, det_label_dir, split='val', class_names=['Car'], frame_format=False):

  if split == 'val':
    sequence_ids = sequence_val_ids
  elif split == 'trainval':
    sequence_ids = sequence_trainval_ids
  else:
    raise NotImplementedError

  eval_det_annos, eval_gt_annos = [], []
  for sdx in tqdm(sequence_ids):
    cur_gt_tracklets = get_track_label_anno(osp.join(gt_label_dir, sdx+'.txt'))
    if not frame_format:
      cur_det_tracklets = get_track_label_anno(osp.join(det_label_dir, sdx+'.txt'))

    frame_ids = np.unique(cur_gt_tracklets['frame_id'])
    for fdx in frame_ids:
      # special case for LiDAR-based 3D detection
      if sdx == '0001' and fdx in [176, 177, 178, 179, 180]:
        continue

      eval_gt_annos.append(dict_select(copy.deepcopy(cur_gt_tracklets), cur_gt_tracklets['frame_id'] == fdx))
      if frame_format:
        img_idx = sdx + '-' + '{:06d}'.format(fdx)
        eval_det_annos.append(get_label_anno(osp.join(det_label_dir, img_idx+'.txt')))
      else:
        eval_det_annos.append(dict_select(copy.deepcopy(cur_det_tracklets), cur_det_tracklets['frame_id'] == fdx))

  # print("3D object evaluate tool from OpenPCDet https://github.com/open-mmlab/OpenPCDet.git")
  ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
  print(ap_result_str)
  # print(ap_dict)


def dict_select(dicts, mask):
  for key, val in dicts.items():
    dicts[key] = val[mask]
  return dicts


def get_track_label_anno(label_path):
  annotations = {}
  annotations.update({
      'frame_id': [],
      'track_id': [],
      'name': [],
      'truncated': [],
      'occluded': [],
      'alpha': [],
      'bbox': [],
      'dimensions': [],
      'location': [],
      'rotation_y': [],
  })
  with open(label_path, 'r') as f:
      lines = f.readlines()
  content = [line.strip().split(' ') for line in lines]

  annotations['frame_id'] = np.array([int(x[0]) for x in content])
  annotations['track_id'] = np.array([int(x[1]) for x in content])
  annotations['name'] = np.array([x[2] for x in content])
  annotations['truncated'] = np.array([float(x[3]) for x in content])
  annotations['occluded'] = np.array([int(float(x[4])) for x in content])
  annotations['alpha'] = np.array([float(x[5]) for x in content])
  annotations['bbox'] = np.array(
      [[float(info) for info in x[6:10]] for x in content]).reshape(-1, 4)
  # dimensions will convert hwl format to standard lhw(camera) format.
  annotations['dimensions'] = np.array(
      [[float(info) for info in x[10:13]] for x in content]).reshape(
          -1, 3)[:, [2, 0, 1]]
  annotations['location'] = np.array(
      [[float(info) for info in x[13:16]] for x in content]).reshape(-1, 3)
  annotations['rotation_y'] = np.array(
      [float(x[16]) for x in content]).reshape(-1)
  if len(content) != 0 and len(content[0]) == 18:  # have score
      annotations['score'] = np.array([float(x[17]) for x in content])
  else:
      annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
  # index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
  # annotations['index'] = np.array(index, dtype=np.int32)
  # annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
  return annotations


def get_label_anno(label_path):
  annotations = {}
  annotations.update({
      'name': [],
      'truncated': [],
      'occluded': [],
      'alpha': [],
      'bbox': [],
      'dimensions': [],
      'location': [],
      'rotation_y': []
  })
  with open(label_path, 'r') as f:
      lines = f.readlines()
  # if len(lines) == 0 or len(lines[0]) < 15:
  #     content = []
  # else:
  content = [line.strip().split(' ') for line in lines]
  num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
  annotations['name'] = np.array([x[0] for x in content])
  num_gt = len(annotations['name'])
  annotations['truncated'] = np.array([float(x[1]) for x in content])
  annotations['occluded'] = np.array([int(float(x[2])) for x in content])
  annotations['alpha'] = np.array([float(x[3]) for x in content])
  annotations['bbox'] = np.array(
      [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
  # dimensions will convert hwl format to standard lhw(camera) format.
  annotations['dimensions'] = np.array(
      [[float(info) for info in x[8:11]] for x in content]).reshape(
          -1, 3)[:, [2, 0, 1]]
  annotations['location'] = np.array(
      [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
  annotations['rotation_y'] = np.array(
      [float(x[14]) for x in content]).reshape(-1)
  if len(content) != 0 and len(content[0]) == 16:  # have score
      annotations['score'] = np.array([float(x[15]) for x in content])
  else:
      annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
  index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
  annotations['index'] = np.array(index, dtype=np.int32)
  annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
  return annotations


if __name__ == '__main__':
  evaluation(args.label_path, args.result_path, split=args.split, class_names=[args.class_name], frame_format=args.framewise)
