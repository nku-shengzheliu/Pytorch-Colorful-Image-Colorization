import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def multiclass_metrics(pred, gt):
  """
  check precision and recall for predictions.
  Output: overall = {precision, recall, f1}
  """
  eps=1e-6
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for ii in range(pred.shape[0]):
    pred_ind = np.array(pred[ii]>0.5, dtype=int)
    gt_ind = np.array(gt[ii]>0.5, dtype=int)
    inter = pred_ind * gt_ind
    # add to overall
    NC += np.sum(inter)
    NP += np.sum(pred_ind)
    NR += np.sum(gt_ind)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall']+eps)
  return overall

def adjust_learning_rate(optimizer, step):
    if step <200000:
        lr = 3*1e-5
    elif step < 375000:
        lr = 1e-5
    else:
        lr = 3*1e-6
    optimizer.param_groups[0]['lr'] = lr