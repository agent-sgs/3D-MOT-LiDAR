import concurrent.futures as futures
import os, glob, copy
import pathlib
import re
from tqdm import tqdm
from collections import OrderedDict, namedtuple

import numpy as np
from skimage import io


def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / union

def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_image_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training,
                               relative_path, exist_check)

def get_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training,
                               relative_path, exist_check)

def get_velodyne_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check)

def get_calib_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check)

def get_kitti_track_path(s_id,
                         prefix,
                         info_type='image_2',
                         file_tail='.png',
                         training=True,
                         relative_path=True,
                         exist_check=True):
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / s_id / file_tail
    else:
        file_path = pathlib.Path('testing') / info_type / s_id / file_tail
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_track_image_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    s_id, i_id = idx.split('-')
    return get_kitti_track_path(s_id, prefix, 'image_02', i_id+'.png', training, relative_path, exist_check)

def get_track_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    s_id, i_id = idx.split('-')
    return get_kitti_track_path(s_id+'.txt', prefix, 'label_02', '', training, relative_path, exist_check)

def get_track_velodyne_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    s_id, i_id = idx.split('-')
    return get_kitti_track_path(s_id, prefix, 'velodyne', i_id+'.bin', training, relative_path, exist_check)

def get_track_calib_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    s_id, i_id = idx.split('-')
    return get_kitti_track_path(s_id+'.txt', prefix, 'calib', '', training, relative_path, exist_check)

def get_track_oxts_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    s_id, i_id = idx.split('-')
    return get_kitti_track_path(s_id+'.txt', prefix, 'oxts', '', training, relative_path, exist_check)

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def dicts_select(dicts, mask):
    for key, value in dicts.items():
        if isinstance(value, dict):
            dicts_select(value, mask)
        else:
            dicts[key] = value[mask]

def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         oxts=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    root_path = pathlib.Path(path)

    def map_func(idx):
        print(idx)
        i_id = int(idx.split('-')[1])
        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_track_velodyne_path(idx, path, training, relative_path)
        image_info['img_path'] = get_track_image_path(idx, path, training, relative_path)
        if with_imageshape:
            img_path = image_info['img_path']
            if relative_path: img_path = str(root_path / img_path)
            image_info['img_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_track_label_path(idx, path, training, relative_path)
            if relative_path: label_path = str(root_path / label_path)
            annotations = get_track_label_anno(label_path)
            dicts_select(annotations, annotations['frame_id'] == i_id)

        if oxts:
            oxts_path = get_track_oxts_path(idx, path, training, relative_path=False)
            pose = PoseGenerator(oxts_path).get(i_id)
            if extend_matrix: pose = _extend_matrix(pose)
            image_info['oxts'] = pose

        if calib:
            calib_path = get_track_calib_path(idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            image_info['calib/P0'] = P0
            image_info['calib/P1'] = P1
            image_info['calib/P2'] = P2
            image_info['calib/P3'] = P3
            R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            image_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
            Tr_imu_to_velo = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
            image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)


def label_str_to_int(labels, remove_dontcare=True, dtype=np.int32):
    class_to_label = get_class_to_label_map()
    ret = np.array([class_to_label[l] for l in labels], dtype=dtype)
    if remove_dontcare:
        ret = ret[ret > 0]
    return ret

def get_class_to_label_map():
    class_to_label = {
        'Car': 0,
        'Pedestrian': 1,
        'Cyclist': 2,
        'Van': 3,
        'Person_sitting': 4,
        'Truck': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1,
    }
    return class_to_label

def get_classes():
    return get_class_to_label_map().keys()

def filter_gt_boxes(gt_boxes, gt_labels, used_classes):
    mask = np.array([l in used_classes for l in gt_labels], dtype=np.bool)
    return mask

def filter_anno_by_mask(image_anno, mask):
    img_filtered_annotations = {}
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][mask])
    return img_filtered_annotations


def filter_infos_by_used_classes(infos, used_classes):
    new_infos = []
    for info in infos:
        annos = info["annos"]
        name_in_info = False
        for name in used_classes:
            if name in annos["name"]:
                name_in_info = True
                break
        if name_in_info:
            new_infos.append(info)
    return new_infos

def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x != "DontCare"
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

def remove_low_height(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['bbox']) if (s[3] - s[1]) >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['score']) if s >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

def keep_arrays_by_name(gt_names, used_classes):
    inds = [
        i for i, x in enumerate(gt_names) if x in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds

def drop_arrays_by_name(gt_names, used_classes):
    inds = [
        i for i, x in enumerate(gt_names) if x not in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds

def apply_mask_(array_dict):
    pass

def filter_kitti_anno(image_anno,
                      used_classes,
                      used_difficulty=None,
                      dontcare_iou=None):
    if not isinstance(used_classes, (list, tuple, np.ndarray)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i for i, x in enumerate(img_filtered_annotations['difficulty'])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = (
                img_filtered_annotations[key][relevant_annotation_indices])

    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations['name'])
            if x == 'DontCare'
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = (img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)])
    return img_filtered_annotations


def filter_annos_class(image_annos, used_class):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['name']) if x in used_class
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_score(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_difficulty(image_annos, used_difficulty):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['difficulty']) if x in used_difficulty
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_height(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['bbox']) if (s[3] - s[1]) >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos

def filter_empty_annos(image_annos):
    new_image_annos = []
    for anno in image_annos:
        if anno["name"].shape[0] > 0:
            new_image_annos.append(anno.copy())
    return new_image_annos


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)

def annos_to_kitti_label(annos):
    num_instance = len(annos["name"])
    result_lines = []
    for i in range(num_instance):
        result_dict = {
            'name': annos["name"][i],
            'truncated': annos["truncated"][i],
            'occluded': annos["occluded"][i],
            'alpha':annos["alpha"][i],
            'bbox': annos["bbox"][i],
            'dimensions': annos["dimensions"][i],
            'location': annos["location"][i],
            'rotation_y': annos["rotation_y"][i],
        }
        line = kitti_result_line(result_dict)
        result_lines.append(line)
    return result_lines

def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    if len(bbox) == 0:
        annos["difficulty"] = np.array([-1], np.int32)
        return -1
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def add_difficulty_to_annos_v2(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = not ((occlusion > max_occlusion[0]) or (height < min_height[0])
                 or (truncation > max_trunc[0]))
    moderate_mask = not ((occlusion > max_occlusion[1]) or (height < min_height[1])
                 or (truncation > max_trunc[1]))
    hard_mask = not ((occlusion > max_occlusion[2]) or (height < min_height[2])
                 or (truncation > max_trunc[2]))
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


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


def get_pseudo_label_anno():
    annotations = {}
    annotations.update({
        'name': np.array(['Car']),
        'truncated': np.array([0.0]),
        'occluded': np.array([0]),
        'alpha': np.array([0.0]),
        'bbox': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'dimensions': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'location': np.array([[0.1, 0.1, 15.0]]),
        'rotation_y': np.array([[0.1, 0.1, 15.0]])
    })
    return annotations

def get_start_result_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations

def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations

def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx_str = get_image_index_str(idx)
        label_filename = label_folder / (image_idx_str + '.txt')
        anno = get_label_anno(label_filename)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos


def get_sequence_label_annos(label_folder, sample_ids=None, test=False):
    annos = []
    test_annos = []
    label_folder = pathlib.Path(label_folder)

    seq_annos = dict()
    for sample_id in tqdm(sample_ids, desc='Loading gt annos ...'):
        s_id, i_id = sample_id.split('-')
        label_file = label_folder / (s_id + '.txt')
        if s_id in seq_annos.keys():
            anno = copy.deepcopy(seq_annos[s_id])
        else:
            anno = get_track_label_anno(label_file)
            del anno['track_id']
            seq_annos[s_id] = copy.deepcopy(anno)
        dicts_select(anno, anno['frame_id'] == int(i_id))

        del anno['frame_id']
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([sample_id] * num_example)
        annos.append(anno)

        if test:
            gt_anno = anno.copy()
            dicts_select(gt_anno, gt_anno['name'] == 'Car')
            gt_anno['score'] += 1
            gt_anno['location'] += 1e-4
            gt_anno['dimensions'] += 1e-5
            test_annos.append(gt_anno)
    if test:
        return annos, test_annos
    return annos


def anno_to_rbboxes(anno):
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

class PoseGenerator(object):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.
    """
    def __init__(self, oxts_path):
        with open(oxts_path, 'r') as f:
            lines = f.readlines()
        self.content = [line.strip().split(' ') for line in lines]

        # Scale for Mercator projection (from first lat value)
        self.scale = None
        # Origin of the global coordinate system (first GPS position)
        self.Tr_0_inv = None

    def __len__(self):
        return len(self.content)

    def get(self, idx):
        line = self.content[idx]
        # Last five entries are flags and counts
        line[:-5] = [float(x) for x in line[:-5]]
        line[-5:] = [int(float(x)) for x in line[-5:]]

        packet = OxtsPacket(*line)
        if self.scale is None:
            self.scale = np.cos(packet.lat * np.pi / 180.)

        R, t = self.pose_from_oxts_packet(packet, self.scale)
        pose = self.transform_from_rot_trans(R, t)
        # if self.Tr_0_inv is None:
        #     self.Tr_0_inv = np.linalg.inv(pose)
        # return self.Tr_0_inv @ pose
        return pose

    def pose_from_oxts_packet(self, packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * \
             np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    def transform_from_rot_trans(self, R, t):
        """Transforation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.hstack([R, t])


if __name__ == '__main__':
    root_path = pathlib.Path('/opt/projects/data/kitti/track/training')
    Pose = PoseGenerator(root_path / 'oxts' / ('0001' + '.txt'))

    all_data = []
    for idx in range(177, 181):
        oxts = Pose.get(idx)
        all_data.append(oxts.reshape((1, 3, 4)))

    all_data = np.vstack(all_data).astype(np.float32)
    np.save(root_path / "0001_177_180.npy", all_data)
    print()