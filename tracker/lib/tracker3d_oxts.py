import numpy as np
from .motion_oxts import KalmanBoxTracker as KalmanBoxTracker
from lib.common_utils.track_utils import hungrarian_association


class Tracker3D(object):
  def __init__(self,
               min_hits=3,
               max_ages=2,
               low_score=0.3,
               high_score=0.6,
               acc_iou_thr=0.1,
               boxes_iou3d_cpu=None):
    """
    max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
    """
    self.max_ages = max_ages
    self.min_hits = min_hits
    self.acc_iou_thr = acc_iou_thr

    self.low_score = low_score
    self.high_score = high_score
    self.boxes_iou3d_cpu = boxes_iou3d_cpu

    self.trajectories = []
    self.frame_count = 0

  def update(self, det_boxes, det_scores, meta):
    """
    det_bboxes: [x y z l h w ry] in velodyne coordinates
    """
    self.frame_count += 1

    """ step 1. tracker prediction
    """
    self.unconf_trajectories = []
    unconf_boxes = [np.empty((0, 7), dtype=np.float32)]
    self.tracked_trajectories = []
    tracked_boxes = [np.empty((0, 7), dtype=np.float32)]

    for trajectory in self.trajectories:
      trajectory.predict()
      pred_bbox3d = trajectory.get_kf_box3d(meta['oxts'])
      if np.any(np.isnan(pred_bbox3d)):
        continue
      if trajectory.hits >= self.min_hits:
        self.tracked_trajectories.append(trajectory)
        tracked_boxes.append(pred_bbox3d)
      else:
        self.unconf_trajectories.append(trajectory)
        unconf_boxes.append(pred_bbox3d)

    unconf_boxes = np.concatenate(unconf_boxes, axis=0)
    tracked_boxes = np.concatenate(tracked_boxes, axis=0)
    trk_bboxes = np.concatenate([tracked_boxes, unconf_boxes], axis=0)  # [x y z l h w ry]
    num_tracked = len(tracked_boxes)

    """ step 3. data association
    """
    trk_ious = self.boxes_iou3d_cpu(det_boxes, trk_bboxes)

    # 1. seperate association
    # tracked_ious = trk_ious[:, :num_tracked]
    # matched_tracked, unmatched_dets, unmatched_tracked = \
    #     hungrarian_association(tracked_ious, iou_thr=self.iou_thr)
    #
    # unmatched_dets = unmatched_dets[unmatched_dets < num_det]
    #
    # unconf_ious = trk_ious[unmatched_dets, num_tracked:]
    # matched_unconf, unm_dets, unmatched_unconf = \
    #     hungrarian_association(unconf_ious, iou_thr=self.iou_thr)
    #
    # matched_unconf[:, 0] = unmatched_dets[matched_unconf[:, 0]]
    # unmatched_dets = unmatched_dets[unm_dets]

    # 2. merged association
    matched_trk, unmatched_dets, unmatched_trk = hungrarian_association(trk_ious, iou_thr=self.acc_iou_thr)
    matched_tracked = matched_trk[matched_trk[:, 1] < num_tracked]
    matched_unconf = matched_trk[matched_trk[:, 1] >= num_tracked]
    matched_unconf[:, 1] -= num_tracked

    """ step 4. update matched tracker, create and initialise new trackers for unmatched detections
    """
    for d, t in matched_tracked:
      self.tracked_trajectories[t].update(det_boxes[d], det_scores[d], meta['oxts'])

    for d, t in matched_unconf:
      self.unconf_trajectories[t].update(det_boxes[d], det_scores[d], meta['oxts'])

    for d in unmatched_dets:
      tracker = KalmanBoxTracker(det_boxes[d], det_scores[d], meta['calib'], meta['oxts'])
      self.unconf_trajectories.append(tracker)

    """ step 5. output and remove dead tracker
    """
    out_ids, out_bboxes, out_scores = self.get_output(meta)

    return out_ids, out_bboxes, out_scores

  def get_output(self, meta):
    out_ids = []
    out_scores = []
    out_bboxes = [np.empty((0, 7), dtype=np.float32)]

    self.trajectories = []
    for trajectory in self.tracked_trajectories:
      if trajectory.no_hit_streak >= self.max_ages:
        continue  # remove dead
      # trajectory.reinitialize()

      # trajectory.update_tracklet_score()
      self.trajectories.append(trajectory)

      out_ids.append(trajectory.id)
      # out_scores.append(trajectory.trk_score)
      out_scores.append(trajectory.det_score)
      out_bboxes.append(trajectory.get_out_box3d(meta['oxts'], self.low_score))
      # out_det_scores.append(tracker.det_score)
      # out_ages.append(tracker.age)

    for trajectory in self.unconf_trajectories:
      if trajectory.no_hit_streak >= self.max_ages:
        continue  # remove deadiou_thr
      # trajectory.update_tracklet_score()
      self.trajectories.append(trajectory)

      if self.frame_count <= self.min_hits or \
         trajectory.hits >= self.min_hits or \
         trajectory.det_score > self.high_score:

        out_ids.append(trajectory.id)
        # out_scores.append(trajectory.trk_score * self.low_score)
        out_scores.append(trajectory.det_score)
        out_bboxes.append(trajectory.get_out_box3d(meta['oxts'], self.low_score))
        # out_det_scores.append(trajectory.det_score)
        # out_ages.append(trajectory.age)
    out_bboxes = np.concatenate(out_bboxes, axis=0)
    return out_ids, out_bboxes, out_scores

  def special_kf_update(self):
    for trajectory in self.trajectories:
      trajectory.kf.predict()
      trajectory.time_since_update = 1

  def special_output(self, meta):
    out_ids = []
    out_scores = []
    out_bboxes = [np.empty((0, 7), dtype=np.float32)]

    for trajectory in self.tracked_trajectories:
      out_ids.append(trajectory.id)
      out_scores.append(trajectory.det_score)
      # out_scores.append(trajectory.trk_score)
      out_bboxes.append(trajectory.get_out_box3d(meta['oxts'], self.low_score))
    out_bboxes = np.concatenate(out_bboxes, axis=0)
    return out_ids, out_bboxes, out_scores

