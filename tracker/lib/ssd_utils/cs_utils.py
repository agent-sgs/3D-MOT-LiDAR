import torch
from lib.ops.iou3d.iou3d_utils import boxes_iou3d_gpu, boxes_iou_bev
from lib.ops.rep3d.rep3d_utils import replace_initial_detections, replace_along_ious


def select_condidate(iboxes3d, iscores, boxes3d, scores, extra_boxes3d, extra_scores, trk_boxes):

  # -- Candidate selection 1. replace extra_boxes3d by initial detections
  if len(iboxes3d) > 0 and len(extra_boxes3d) > 0:

    # 1.
    # iscores = torch.cat((iscores, extra_scores))
    # iscores, sorted_indices = torch.sort(iscores, descending=False)
    #
    # iboxes3d = torch.cat((iboxes3d, extra_boxes3d), dim=0)
    # iboxes3d = iboxes3d[sorted_indices, :]
    #
    # extra_overs = boxes_iou3d_gpu(extra_boxes3d, iboxes3d)
    # extra_overs = extra_overs > 0
    # extra_scores, extra_boxes3d = replace_initial_detections(extra_overs.bool(), iscores, iboxes3d)

    # 2.
    extra_overs = boxes_iou3d_gpu(extra_boxes3d, iboxes3d)
    max_overs, indices = extra_overs.max(dim=1)
    rep_mask = (max_overs > 0) & (extra_scores < iscores[indices])
    extra_boxes3d[rep_mask] = iboxes3d[indices[rep_mask]]
    extra_scores[rep_mask] = iscores[indices[rep_mask]]

    # 3.
    # extra_overs = boxes_iou3d_gpu(extra_boxes3d, iboxes3d)
    # extra_overs, sorted_indices = torch.sort(extra_overs, dim=1, descending=True)
    # extra_scores, extra_boxes3d = replace_along_ious(extra_overs, sorted_indices, iscores, iboxes3d,
    #                                                  extra_scores, extra_boxes3d)

  # -- Candidate selection 2. directly remove extra_boxes overlapped with final det_boxes
  if len(boxes3d) > 0 and len(extra_boxes3d) > 0:
    extra_overs = boxes_iou3d_gpu(extra_boxes3d, boxes3d)
    rm_mask = torch.logical_not(torch.any(extra_overs > 0, dim=1))
    extra_boxes3d = extra_boxes3d[rm_mask]
    extra_scores = extra_scores[rm_mask]

  cand_boxes = boxes3d
  cand_scores = scores
  if len(extra_boxes3d) > 0:
    cand_boxes = torch.cat((cand_boxes, extra_boxes3d), dim=0)
    cand_scores = torch.cat((cand_scores, extra_scores))

  trk_ious = boxes_iou3d_gpu(cand_boxes, trk_boxes)

  return len(boxes3d), trk_ious.cpu().numpy(), cand_boxes.cpu().numpy(), cand_scores.cpu().numpy()