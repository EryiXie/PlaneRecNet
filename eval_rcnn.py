"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/data/config.py
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import random
import os
import time
from collections import OrderedDict
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from planerecnet import PlaneRecNet
from models.functions.funcs import bbox_iou, mask_iou
from data.datasets import PlaneAnnoDataset, detection_collate, ScanNetDataset
from data.config import set_cfg, set_dataset, cfg, MEANS
from data.augmentations import BaseTransform
from utils.utils import MovingAverage, ProgressBar, SavePath
from utils import timer
from simple_inference import display_on_frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='PlaneRecNet Evaluation')
    parser.add_argument('--trained_model',
                        default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    global args
    args = parser.parse_args(argv)


semantic_metrics = ["RI", " VOI" , "SC"]
depth_metrics = ["abs_rel", "sq_rel", "rmse", "log10", "a1", "a2", "a3", "ratio"]
iou_thresholds = [x / 100 for x in range(50, 100, 5)]

def evaluate(dataset, during_training=False, eval_nums=-1):
    eval_nums = len(dataset) - 1 if eval_nums < 0 else min(eval_nums, len(dataset))
    progress_bar = ProgressBar(30, eval_nums)

    print()

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    dataset_indices = dataset_indices[:eval_nums]

    infos = []
    ap_data = {
        'box': [APDataObject()  for _ in iou_thresholds],
        'mask': [APDataObject() for _ in iou_thresholds]
    }

    ROOT = "/home/xie/Documents/train_history/vis_results/scannet/planeae"

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):

            image, gt_instances, gt_depth = dataset.pull_item(image_idx)
            img_id = dataset.ids[image_idx]
            file_name_raw = dataset.coco.loadImgs(img_id)[0]['file_name']
            file_name = file_name_raw.split('/')[0] + "_" + file_name_raw.split('/')[-1]
            depth_path = os.path.join(ROOT, "dep" ,file_name.replace('.jpg', '.png'))
            seg_path = os.path.join(ROOT, "seg" ,file_name.replace('.jpg', '.npy'))
            #print(file_name, file_name_raw, depth_path, seg_path)
            pred_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float) / 512
            pred_depth = torch.from_numpy(pred_depth).unsqueeze(dim=0).cuda()

            pred_masks = torch.from_numpy(np.load(seg_path)).cuda()
            pred_boxes = torch.zeros(pred_masks.size(0), 4).cuda()
            for i in range(pred_masks.size(0)):
                    mask = pred_masks[i].squeeze()
                    ys, xs = torch.where(mask)
                    #print(ys.shape, xs.shape)
                    if ys.shape == torch.Size([0]) or xs.shape == torch.Size([0]):
                        pred_boxes[i] = torch.tensor([0, 0, 0, 0]).float()
                    else:
                        pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
            pred_classes = torch.zeros(pred_masks.shape[0]).cuda()
            scr_path = os.path.join(ROOT, "seg" ,file_name.replace('.jpg', '_score.npy'))
            #pred_scores = torch.from_numpy(np.load(scr_path)).cuda()
            pred_scores = torch.ones_like(pred_classes).cuda() * 0.8 + torch.rand_like(pred_classes).cuda()*0.2
            #pred_scores = torch.rand_like(pred_classes).cuda()
            gt_masks, gt_boxes, gt_classes, gt_planes, k_matrices = [v.cuda() for k, v in gt_instances.items()]


            gt_depth = gt_depth.cuda()
            depth_error_per_frame = compute_depth_metrics(pred_depth, gt_depth, median_scaling=True)
            infos.append(depth_error_per_frame)

            if pred_masks is not None:   
                pred_masks = pred_masks.float()
                gt_masks = gt_masks.float()
                compute_segmentation_metrics(ap_data, gt_masks, gt_boxes, gt_classes, pred_masks, pred_boxes, pred_classes, pred_scores)
            

            if not args.no_bar:
                progress = (it+1) / eval_nums * 100
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%) '
                      % (repr(progress_bar), it+1, eval_nums, progress), end='')
        
        calc_map(ap_data)
        infos = np.asarray(infos, dtype=np.double)
        infos = infos.sum(axis=0)/infos.shape[0]
        print("Depth Metrics:")
        print("{}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f} \n{}: {:.5f}".format(
            depth_metrics[0], infos[0], depth_metrics[1], infos[1], depth_metrics[2], infos[2],
            depth_metrics[3], infos[3], depth_metrics[4], infos[4], depth_metrics[5], infos[5],
            depth_metrics[6], infos[6], depth_metrics[7], infos[7]
        ))
        
            

    except KeyboardInterrupt:
        print('Stopping...')


def compute_depth_metrics(pred_depth, gt_depth, median_scaling=True):
    """
    Computation of error metrics between predicted and ground truth depths.
    Prediction and ground turth need to be converted to the same unit e.g. [meter].

    Arguments: pred_depth, gt_depth: Tensor [1, H, W], dense depth map
               median_scaling: If True, use median value to scale pred_depth
    Returns: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3: depth metrics
             ratio: median ration between pred_depth and gt_depth, if not median_scaling, ratio = 0
    """
    _, H, W = gt_depth.shape
    pred_depth_flat = pred_depth.squeeze().view(-1, H*W)
    gt_depth_flat = gt_depth.squeeze().view(-1, H*W)
    #print(gt_depth_flat.min(), gt_depth_flat.max())
    valid_mask = (gt_depth_flat > 0.5).logical_and(pred_depth_flat > 0.5)
    pred_depths_flat = pred_depth_flat[valid_mask]
    gt_depths_flat = gt_depth_flat[valid_mask]

    if median_scaling:
        ratio = torch.median(gt_depth) / torch.median(pred_depth)
        pred_depth *= ratio
    else:
        ratio = 0

    pred_depth[pred_depth < cfg.dataset.min_depth] = cfg.dataset.min_depth
    pred_depth[pred_depth > cfg.dataset.max_depth] = cfg.dataset.max_depth

    thresh = torch.max((gt_depths_flat / pred_depths_flat), (pred_depths_flat / gt_depths_flat))
    a1 = (thresh < 1.25     ).type(torch.cuda.DoubleTensor).mean()
    a2 = (thresh < 1.25 ** 2).type(torch.cuda.DoubleTensor).mean()
    a3 = (thresh < 1.25 ** 3).type(torch.cuda.DoubleTensor).mean()

    rmse = (gt_depths_flat - pred_depths_flat) ** 2
    rmse = torch.sqrt(rmse.mean())

    #rmse_log = (torch.log(gt_depths_flat) - torch.log(pred_depths_flat)) ** 2
    #rmse_log = torch.sqrt(rmse_log.mean())

    log10 = torch.mean(torch.abs(torch.log10(gt_depths_flat) - torch.log10(pred_depths_flat)))

    abs_rel = torch.mean(torch.abs(gt_depths_flat - pred_depths_flat) / gt_depths_flat)
    sq_rel = torch.mean(((gt_depths_flat - pred_depths_flat) ** 2) / gt_depths_flat)

    return abs_rel.cpu(), sq_rel.cpu(), rmse.cpu(), log10.cpu(), a1.cpu(), a2.cpu(), a3.cpu(), ratio.cpu()


def compute_segmentation_metrics(ap_data, gt_masks, gt_boxes, gt_classes, pred_masks, pred_boxes, pred_classes, pred_scores):
    num_pred = len(pred_classes)
    num_gt   = len(gt_classes)

    indices = sorted(range(num_pred), key=lambda i: -pred_masks[i].sum())
    pred_masks = pred_masks[indices]
    pred_boxes = pred_boxes[indices]
    pred_classes = pred_classes[indices]
    pred_scores = pred_scores[indices]

    idxx = sorted(range(gt_masks.shape[0]), key=lambda i: -gt_masks[i].sum())
    gt_masks = gt_masks[idxx]
    gt_boxes = gt_boxes[idxx]
    gt_classes = gt_classes[idxx]



    # TODO: Take a look at the float and cpu flages
    mask_iou_cache = mask_iou(pred_masks, gt_masks).cpu()
    bbox_iou_cache = bbox_iou(pred_boxes.float(), gt_boxes.float()).cpu()

    #indices = sorted(range(num_pred), key=lambda i: -pred_scores[i])
    indices = sorted(range(num_pred), key=lambda i: -pred_masks[i].sum())


    iou_types = [
        ('box', lambda i, j: bbox_iou_cache[i, j].item(),
        lambda i: pred_scores[i], indices),
        ('mask', lambda i, j: mask_iou_cache[i, j].item(),
        lambda i: pred_scores[i], indices)
    ]

    ap_per_iou = []
    num_gt_for_class = sum([1 for x in gt_classes if x == 0])
    for iouIdx in range(len(iou_thresholds)):
        iou_threshold = iou_thresholds[iouIdx]
        for iou_type, iou_func, score_func, indices in iou_types:
            gt_used = [False] * len(gt_classes)
            ap_obj = ap_data[iou_type][iouIdx]
            ap_obj.add_gt_positives(num_gt_for_class)

            for i in indices:
                max_iou_found = iou_threshold
                max_match_idx = -1
                for j in range(num_gt):
                    iou = iou_func(i, j)
                    if iou > max_iou_found:
                        max_iou_found = iou
                        max_match_idx = j

                if max_match_idx >= 0:
                    gt_used[max_match_idx] = True
                    ap_obj.push(score_func(i), True)
                
                ap_obj.push(score_func(i), False)


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        y_range = [0] * 101
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for iou_idx in range(len(iou_thresholds)):
        for iou_type in ('box', 'mask'):
            ap_obj = ap_data[iou_type][iou_idx]
            if not ap_obj.is_empty():
                aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * \
                100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (
            sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()}
                for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    def make_row(vals): return (' %5s |' * len(vals)) % tuple(vals)
    def make_sep(n): return ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int)
                            else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' %
                                     x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


if __name__ == '__main__':
    import datetime

    parse_args()
    
    if args.dataset is not None:
        set_dataset(args.dataset)
    
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')
        
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        dataset = ScanNetDataset(cfg.dataset.valid_images, cfg.dataset.valid_info,transform=BaseTransform(MEANS), has_gt=cfg.dataset.has_gt, has_pos=cfg.dataset.has_pos)
        evaluate(dataset, during_training=False, eval_nums=args.max_images)
