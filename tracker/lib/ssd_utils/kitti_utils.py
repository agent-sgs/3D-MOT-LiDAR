from copy import deepcopy
import numpy as np
from lib.ssd_utils.geometry import center_to_corner_box3d


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def read_track_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Tracklet3d(line) for line in lines]
    return objects

def read_lidar(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj


def get_detections_from_label_file(label_file, calib, class_names=['Car']):
  objects = read_label(label_file)
  det_bboxes = [object.box3d for object in objects if object.type in class_names]

  det_bboxes = np.array(det_bboxes, dtype=np.float64)
  det_scores = np.array([object.score for object in objects if object.type in class_names], dtype=np.float64)

  if len(det_bboxes) == 0:
    return np.zeros((0, 7), dtype=np.float64), np.zeros(0, dtype=np.float64)
  return det_bboxes, det_scores


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.loc = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        try:
            self.score = data[15]
        except:
            self.score = 1.
        self.box3d = np.array([data[11], data[12], data[13], data[9], data[10], data[8], data[14]]).astype(np.float32)
    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.loc[0], self.loc[1], self.loc[2], self.ry))


class Tracklet3d(object):
    ''' 3d trajectory label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[4:] = [float(x) for x in data[4:]]

        self.frame_id = int(data[0])
        self.track_id = int(data[1])
        # extract label, truncation, occlusion
        self.type = data[2]  # 'Car', 'Pedestrian', ...
        self.truncation = data[3]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[4])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[5]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[6]  # left
        self.ymin = data[7]  # top
        self.xmax = data[8]  # right
        self.ymax = data[9]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[10]  # box height
        self.w = data[11]  # box width
        self.l = data[12]  # box length (in meters)
        self.loc = (data[13], data[14], data[15])  # location (x,y,z) in camera coord.
        self.ry = data[16]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        try:
            self.score = data[17]
        except:
            self.score = 1.
        self.box3d = np.array([data[13], data[14], data[15], data[11], data[12], data[10], data[16]]).astype(np.float64)
    def print_tracklet(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.loc[0], self.loc[1], self.loc[2], self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3, 4])

        # Projection matrix from rect camera coord to image3 coord
        self.P3 = calibs['P3']
        self.P3 = np.reshape(self.P3, [3, 4])

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])

        self.C2V = np.zeros_like(self.V2C)  # 3x4
        self.C2V[0:3, 0:3] = np.transpose(self.V2C[0:3, 0:3])
        self.C2V[0:3, 3] = np.dot(-np.transpose(self.V2C[0:3, 0:3]), \
                                  self.V2C[0:3, 3])

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P2[0, 2]
        self.c_v = self.P2[1, 2]
        self.f_u = self.P2[0, 0]
        self.f_v = self.P2[1, 1]
        self.b_x = self.P2[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P2[1, 3] / (-self.f_v)

        if from_video:
            self.I2V = np.reshape(calibs['Tr_imu_to_velo'], [3, 4])
            self.I2V_ext = np.vstack((self.I2V, np.array([[0., 0., 0., 1.]])))
            V2C_ext = np.vstack((self.V2C, np.array([[0., 0., 0., 1.]])))

            R0_ext = np.hstack((np.vstack((self.R0, np.array([[0., 0., 0.]]))),
                                np.array([[0], [0], [0], [1]])))
            self.I2R_ext = np.dot(np.dot(R0_ext, V2C_ext), self.I2V_ext)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_filepath):
        ''' Read calibration for camera 2 from 3D MOT trajectories '''
        with open(calib_filepath, 'r') as f:
            lines = f.readlines()
        lines = [line.split(' ') for line in lines]

        data = {}
        data['P0'] = np.array([float(x) for x in lines[0][1:13]], dtype=np.float64)
        data['P1'] = np.array([float(x) for x in lines[1][1:13]], dtype=np.float64)
        data['P2'] = np.array([float(x) for x in lines[2][1:13]], dtype=np.float64)
        data['P3'] = np.array([float(x) for x in lines[3][1:13]], dtype=np.float64)

        data['R0_rect'] = np.array([float(x) for x in lines[4][1:10]], dtype=np.float64)
        data['Tr_velo_to_cam'] = np.array([float(x) for x in lines[5][1:13]], dtype=np.float64)
        data['Tr_imu_to_velo'] = np.array([float(x) for x in lines[6][1:13]], dtype=np.float64)
        return data

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = list(pts_3d.shape[0:-1])
    pts_3d_hom = np.concatenate([pts_3d, np.ones(n + [1], dtype=np.float64)], axis=-1)
    return pts_3d_hom

# ===========================
# ------- 3d to 3d ----------
# ===========================
def project_velo_to_ref(pts_3d_velo, calib):
    pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
    return pts_3d_velo @ calib.V2C.T

def project_ref_to_velo(pts_3d_ref,calib):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return pts_3d_ref @ calib.C2V.T

def project_rect_to_ref(pts_3d_rect, calib):
    ''' Input and Output are nx3 points '''
    return pts_3d_rect @ np.linalg.inv(calib.R0).T
    #return np.transpose(np.dot(np.linalg.inv(calib.R0), np.transpose(pts_3d_rect)))

def project_ref_to_rect(pts_3d_ref, calib):
    ''' Input and Output are nx3 points '''
    return pts_3d_ref @ calib.R0.T

def project_velo_to_rect(pts_3d_velo, calib):
    pts_3d_ref = project_velo_to_ref(pts_3d_velo, calib)
    return project_ref_to_rect(pts_3d_ref, calib)

def project_rect_to_velo(pts_3d_rect, calib):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, calib)
    return project_ref_to_velo(pts_3d_ref, calib)


def box3d_world_to_lidar(box3d_world, calib, W2I):
  box3d = deepcopy(box3d_world)
  pts_hom = np.ones((4, 1))
  pts_hom[:3, 0] = box3d_world[:3]

  W2I = np.dot(calib.I2V, W2I)
  box3d[:3] = np.dot(W2I, pts_hom)[:3, 0]

  pts_hom[0] += np.cos(box3d_world[6]) * 0.1
  pts_hom[1] += np.sin(box3d_world[6]) * 0.1
  pts_hom = np.dot(W2I, pts_hom)[:3, 0]
  box3d[6] = np.arctan2(pts_hom[1] - box3d[1],
                        pts_hom[0] - box3d[0])
  return box3d


def box3d_lidar_to_world(box3d, calib, W2I):
  box3d_world = deepcopy(box3d)
  pts_hom = np.ones((4, 1))
  pts_hom[:3, 0] = box3d[:3]

  V2W = np.linalg.inv(np.dot(calib.I2V_ext, W2I))
  box3d_world[:3] = np.dot(V2W, pts_hom)[:3, 0]

  pts_hom[0] += np.cos(box3d[6]) * 0.1
  pts_hom[1] += np.sin(box3d[6]) * 0.1
  pts_hom = np.dot(V2W, pts_hom)[:3]
  box3d_world[6] = np.arctan2(pts_hom[1] - box3d_world[1],
                              pts_hom[0] - box3d_world[0])
  return box3d_world


# ===========================
# ------- 3d to 2d ----------
# ===========================
def project_rect_to_image(pts_3d_rect, calib):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = pts_3d_rect @ calib.P2.T
    pts_2d[..., 0] /= pts_2d[..., 2]
    pts_2d[..., 1] /= pts_2d[..., 2]
    return pts_2d[..., 0:2]

def project_velo_to_image(pts_3d_velo, calib):
    ''' Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = project_velo_to_rect(pts_3d_velo, calib)
    return project_rect_to_image(pts_3d_rect, calib)

def project_rect_to_right(pts_3d_rect, calib):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = pts_3d_rect @ calib.P3.T
    pts_2d[..., 0] /= pts_2d[..., 2]
    pts_2d[..., 1] /= pts_2d[..., 2]
    return pts_2d[..., 0:2]

def project_velo_to_right(pts_3d_velo, calib):
    ''' Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = project_velo_to_rect(pts_3d_velo, calib)
    return project_rect_to_right(pts_3d_rect, calib)

# ===========================
# ------- 2d to 3d ----------
# ===========================
def project_image_to_rect(uv_depth, calib):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calib.c_u) * uv_depth[:, 2]) / calib.f_u + calib.b_x
    y = ((uv_depth[:, 1] - calib.c_v) * uv_depth[:, 2]) / calib.f_v + calib.b_y
    pts_3d_rect = np.zeros((n, 3), dtype=np.float64)
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def project_image_to_velo(uv_depth, calib):
    pts_3d_rect = project_image_to_rect(uv_depth, calib)
    return project_rect_to_velo(pts_3d_rect, calib)


def limit_period(val, offset=0.5, period=np.pi):
  return val - np.floor(val / period + offset) * period


def kitti_bbox2results(frame, pred_boxes, pred_scores, pred_labels, track_ids, meta, eval_file):
  """
  format for dimensions: boxes_lidar: l w h -> boxes_cam: l h w -> label_file - h w l
  """
  if len(track_ids) == 0:
    return None

  calib = meta['calib']
  image_shape = meta['img_shape']

  pred_boxes[:, 6] = limit_period(pred_boxes[:, 6], 0.5, 2 * np.pi)
  pred_boxes_camera = np.zeros_like(pred_boxes)
  pred_boxes_camera[:, :3] = project_velo_to_rect(pred_boxes[:, :3], calib)
  pred_boxes_camera[:, 3:] = pred_boxes[:, [4, 5, 3, 6]]

  corners_camera = center_to_corner_box3d(pred_boxes_camera, origin=[0.5, 1.0, 0.5], axis=1)
  corners_rgb = project_rect_to_image(corners_camera, calib)

  minxy = np.min(corners_rgb, axis=1)
  maxxy = np.max(corners_rgb, axis=1)
  pred_boxes_img = np.concatenate([minxy, maxxy], axis=1)

  pred_alphas = -np.arctan2(-pred_boxes_camera[:, 1], pred_boxes_camera[:, 0]) + pred_boxes_camera[:, 6]

  template = '{:d} ' + '{:d} ' + '{} ' + '0 0 ' + ' '.join(['{:.4f}' for _ in range(13)]) + '\n'
  for bbox2d, bbox3d, type, score, alpha, track_id in \
      zip(pred_boxes_img, pred_boxes_camera, pred_labels, pred_scores, pred_alphas, track_ids):
    if bbox2d[0] > image_shape[1] or bbox2d[1] > image_shape[0]:
      continue
    if bbox2d[2] < 0 or bbox2d[3] < 0:
      continue

    bbox2d[0] = np.clip(bbox2d[0], a_min=0, a_max=image_shape[1] - 1)
    bbox2d[1] = np.clip(bbox2d[1], a_min=0, a_max=image_shape[0] - 1)
    bbox2d[2] = np.clip(bbox2d[2], a_min=0, a_max=image_shape[1] - 1)
    bbox2d[3] = np.clip(bbox2d[3], a_min=0, a_max=image_shape[0] - 1)

    dim = bbox3d[[3, 4, 5]]  # l h w
    loc = bbox3d[:3]  # x y z in camera coordinate

    line = template.format(frame, track_id, type, alpha, *bbox2d, *dim[[1, 2, 0]], *loc, bbox3d[6], score)
    eval_file.write(line)

