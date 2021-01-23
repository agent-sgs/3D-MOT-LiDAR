import numpy as np
from copy import deepcopy
from filterpy.kalman import KalmanFilter
from lib.common_utils.box_utils import box3d_lidar_to_world, box3d_world_to_lidar


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class KalmanBoxTracker(object):
    """
    	This class represents the internel state of individual tracked objects observed as bbox.
    	"""
    count = 0
    def __init__(self, box3D, score, calib, oxts):
        """
        Initialises a tracker using initial bounding box. dt = 0.1
        state: [x vx, ax, y, vy, ay, z, vz, az, r, vr, ar, l, w, h]
        origin coordinate defined at starting point, namely, bbox3d == bbox3d_world for initial point
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.age = 1
        self.hits = 1
        self.hit_streak = 1
        self.no_hit_streak = 0  # time_since_update

        self.det_streak  = 4  # param for reinilialization, 4 - avoid initializing at first interval
        self.det_score = score
        self.det_count = 1
        self.det_score_sum = score
        self.trk_score = score

        self.calib = calib
        self.oxts_init = oxts
        box3D_world = box3d_lidar_to_world(box3D, calib, np.eye(4))

        # boxes_world = boxes3d_lidar_to_world(box3D, calib, np.eye(4))
        # boxes = boxes3d_world_to_lidar(boxes_world, calib, np.eye(4))
        # box3D = box3d_world_to_lidar(box3D_world.reshape(7), calib, np.eye(4))

        self.kf_init(box3D_world)

        self.history_meta = {'box3D': [box3D], 'oxts': [oxts]}

    def kf_init(self, box3D_world, dt=1):
      # define constant acceleration model
      self.kf = KalmanFilter(dim_x=13, dim_z=7)
      self.kf.F = np.array([[1, dt, .5 * dt * dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # state transition matrix
                            [0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, dt, .5 * dt * dt, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, dt, .5 * dt * dt, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

      self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

      self.kf.P *= 10.
      self.kf.P[[1, 2, 4, 5, 7, 8], [1, 2, 4, 5, 7, 8]] *= 1000.
      self.kf.Q[[1, 2, 4, 5, 7, 8], [1, 2, 4, 5, 7, 8]] *= 0.01

      self.kf.x[[0, 3, 6, 9, 10, 11, 12]] = box3D_world.reshape((7, 1))

    def kf_update(self, box3D_world):

        self.kf.x[12] = limit_period(self.kf.x[12] + np.pi, 0.5, 2 * np.pi)
        acute_angle = abs(box3D_world[6] - self.kf.x[12])
        if acute_angle > np.pi * 0.5 and acute_angle < np.pi * 1.5:
            self.kf.x[12] = limit_period(self.kf.x[12] + np.pi, 0.5, 2 * np.pi)

        if abs(box3D_world[6] - self.kf.x[12]) >= np.pi * 1.5:
            self.kf.x[12] += 2 * np.pi * np.sign(box3D_world[6])

        self.kf.update(box3D_world)

    def update(self, box3D, score, oxts):
        """
        Updates the state vector with observed bbox.
        """
        box3D_world = box3d_lidar_to_world(box3D, self.calib, np.dot(np.linalg.inv(oxts), self.oxts_init))

        # boxes_world = boxes3d_lidar_to_world(box3D, self.calib, np.dot(np.linalg.inv(oxts), self.oxts_init))
        # boxes = boxes3d_world_to_lidar(boxes_world, self.calib, np.dot(np.linalg.inv(oxts), self.oxts_init))
        # box3D = box3d_world_to_lidar(box3D_world.reshape(7), self.calib, np.dot(np.linalg.inv(oxts), self.oxts_init))

        self.kf_update(box3D_world)
        self.det_score = score
        self.det_count += 1
        self.det_score_sum += score

        self.history_meta['box3D'].append(box3D)
        self.history_meta['oxts'].append(oxts)

        self.hits += 1
        self.hit_streak += 1
        self.det_streak += 1
        self.no_hit_streak = 0

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()

        self.age += 1
        if self.no_hit_streak > 0:
            self.hit_streak = 0
            self.det_streak = 0  # param for reinitialization
        self.no_hit_streak += 1

    def get_kf_box3d(self, oxts):
        box3D_world = self.kf.x[[0, 3, 6, 9, 10, 11, 12]].reshape(7,)
        box3D = box3d_world_to_lidar(box3D_world, self.calib, np.dot(np.linalg.inv(oxts), self.oxts_init))
        return box3D.reshape(1, 7)

    def get_out_box3d(self, oxts, score_thr=0.1):
        """ return predicted bbox3d_world, transformed to current velodyne coordinates
        """
        if self.no_hit_streak == 0 and self.det_score > score_thr:
            return (self.history_meta['box3D'][-1]).reshape((1, 7))
        return self.get_kf_box3d(oxts)

    def update_tracklet_score(self):
        avg_score = self.det_score_sum / self.det_count

        if self.no_hit_streak == 0:
            self.trk_score = max(0, min(1, avg_score * (np.power(1.1, self.hit_streak) - 1) + self.det_score))
        else:
            self.trk_score = max(0, self.trk_score - np.log(1 + self.no_hit_streak))
            self.det_score = max(0, self.det_score - np.log(1 + self.no_hit_streak))

    def reinitialize(self):
      def update_history(history_meta):
        history_meta['box3D'] = history_meta['box3D'][-3:]
        history_meta['oxts'] = history_meta['oxts'][-3:]
        return history_meta

      if self.det_streak == 3:
        self.history_meta = update_history(self.history_meta)
        self.oxts_init = deepcopy(self.history_meta['oxts'][0])
        box3D_world = box3d_lidar_to_world(self.history_meta['box3D'][0], self.calib, np.eye(4))
        self.kf_init(box3D_world)

        for box3D, oxts in zip(self.history_meta['box3D'][1:], self.history_meta['oxts'][1:]):
          self.kf.predict()

          box3D_world = box3d_lidar_to_world(box3D, self.calib, np.dot(np.linalg.inv(oxts), self.oxts_init))
          self.kf_update(box3D_world)

