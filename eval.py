"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/data/config.py
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import random
import os
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
from data.datasets import PlaneAnnoDataset, detection_collate, ScanNetDataset, NYUDataset
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
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--score_threshold', default=0.15, type=float, 
                        help='Detections with a score under this threshold will not be considered.')
    parser.add_argument("--nms_mode", default="matrix", type=str, choices=["matrix", "mask"], help='Chose NMS type from matrix and mask nms.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--autopsy', dest='autopsy', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    global args
    args = parser.parse_args(argv)

depth_metrics = ["abs_rel", "sq_rel", "rmse", "log10", "a1", "a2", "a3", "ratio"]
iou_thresholds = [x / 100 for x in range(50, 100, 5)]

def evaluate(net: PlaneRecNet, dataset, during_training=False, eval_nums=-1):
    frame_times = MovingAverage()
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

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            image, gt_instances, gt_depth = dataset.pull_item(image_idx)
            batch = Variable(image.unsqueeze(0)).cuda()

            batched_result = net(batch) # if batch_size = 1, result = batched_result[0]
            result = batched_result[0]

            # TODO: this dict looping is not a good practice, python < 3.6 doesn't keep keys/values in same order as declared.
            gt_masks, gt_boxes, gt_classes, gt_planes, k_matrices = [v.cuda() for k, v in gt_instances.items()]
            pred_masks, pred_boxes, pred_classes, pred_scores, pred_depth = [v for k, v in result.items()]

            gt_depth = gt_depth.cuda()
            depth_error_per_frame = compute_depth_metrics(pred_depth, gt_depth, median_scaling=True)
            infos.append(depth_error_per_frame)

            if pred_masks is not None:
                pred_masks = pred_masks.float()
                gt_masks = gt_masks.float()
                compute_segmentation_metrics(ap_data, gt_masks, gt_boxes, gt_classes, pred_masks, pred_boxes, pred_classes, pred_scores)
            
            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())

            if not args.no_bar:
                if it > 1:
                    fps = 1000 / frame_times.get_avg()
                else:
                    fps = 0
                progress = (it+1) / eval_nums * 100
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it+1, eval_nums, progress, fps), end='')
        calc_map(ap_data)
        infos = np.asarray(infos, dtype=np.double)
        infos = infos.sum(axis=0)/infos.shape[0]
        print()
        print("Depth Metrics:")
        print("{}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f} \n{}: {:.5f}".format(
            depth_metrics[0], infos[0], depth_metrics[1], infos[1], depth_metrics[2], infos[2],
            depth_metrics[3], infos[3], depth_metrics[4], infos[4], depth_metrics[5], infos[5],
            depth_metrics[6], infos[6], depth_metrics[7], infos[7]
        ))

    except KeyboardInterrupt:
        print('Stopping...')

def tensorborad_visual_log(net: PlaneRecNet, dataset, writer: SummaryWriter, iteration, eval_nums):
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    dataset_indices = dataset_indices[:eval_nums]

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            image, _, _ = dataset.pull_item(image_idx)
            frame_ori = dataset.pull_image(image_idx)
            frame_tensor = torch.from_numpy(frame_ori).cuda().float()
            batch = Variable(image.unsqueeze(0)).cuda()

            batched_result = net(batch) # if batch_size = 1, result = batched_result[0]
            seg_on_frame_numpy, pred_depth = display_on_frame(batched_result[0], frame_tensor, mask_alpha=0.35)

            pred_depth = pred_depth[20:460,20:620]
            vmin = np.percentile(pred_depth, 1)
            vmax = np.percentile(pred_depth, 99)
            pred_depth = pred_depth.clip(min=vmin, max=vmax)
            pred_depth = ((pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) * 255).astype(np.uint8)
            pred_depth_color = cv2.applyColorMap(pred_depth, cv2.COLORMAP_VIRIDIS)

            pred_depth_color = cv2.cvtColor(pred_depth_color, cv2.COLOR_BGR2RGB)
            seg_on_frame_numpy = cv2.cvtColor(seg_on_frame_numpy, cv2.COLOR_BGR2RGB)
            writer.add_image("depth/pred/{}".format(it), pred_depth_color, iteration, dataformats='HWC')
            writer.add_image("seg/pred/{}".format(it), seg_on_frame_numpy, iteration, dataformats='HWC')

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
    valid_mask = (gt_depth_flat > 0.5).logical_and(pred_depth_flat > 0.5)
    pred_depths_flat = pred_depth_flat[valid_mask]
    gt_depths_flat = gt_depth_flat[valid_mask]

    if median_scaling:
        # just to calculate the ratio, we don'r really use median scaling to align pred and gt.
        ratio = torch.median(gt_depth) / torch.median(pred_depths_flat)
        #pred_depths_flat *= ratio
    else:
        ratio = 0

    pred_depths_flat[pred_depths_flat < cfg.dataset.min_depth] = cfg.dataset.min_depth
    pred_depths_flat[pred_depths_flat > cfg.dataset.max_depth] = cfg.dataset.max_depth

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

    mask_iou_cache = mask_iou(pred_masks, gt_masks).cpu()
    bbox_iou_cache = bbox_iou(pred_boxes.float(), gt_boxes.float()).cpu()

    indices = sorted(range(num_pred), key=lambda i: -pred_scores[i])

    iou_types = [
        ('box', lambda i, j: bbox_iou_cache[i, j].item(),
        lambda i: pred_scores[i], indices),
        ('mask', lambda i, j: mask_iou_cache[i, j].item(),
        lambda i: pred_scores[i], indices)
    ]

    ap_per_iou = []

    # THAT THE LINE THAT COMPELETELY WRONG, which used to be: num_gt_for_class = 1
    # num_gt_for_class is not "numbers of classes in gt", it is NUMBERS OF GT INSTANCES OF ONE SINGLE CLASS IN ONE INPUT IMAGE!
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

    new_nms_config = {
        'nms_type': args.nms_mode, 
        'mask_thr': args.score_threshold, 
        'update_thr': args.score_threshold,
        'top_k': args.top_k,}

    set_cfg(args.config)
    cfg.solov2.replace(new_nms_config)

    if args.config is not None:
        set_cfg(args.config)
    
    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)
    
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)
    
    if args.dataset is not None:
        set_dataset(args.dataset)
    
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')
        
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        dataset = eval(cfg.dataset.name)(cfg.dataset.eval_images, cfg.dataset.eval_info,transform=BaseTransform(MEANS), has_gt=cfg.dataset.has_gt, has_pos=cfg.dataset.has_pos)
        print("Loading model...", end='')
        net = PlaneRecNet(cfg)
        net.load_weights(args.trained_model)
        net.eval()
        print("done.")

        net = net.cuda()
        evaluate(net, dataset, during_training=False, eval_nums=args.max_images)

        if args.autopsy:
            begin_time = (datetime.datetime.now()).strftime("%d%m%Y%H%M%S")
            logpath = os.path.join(args.log_folder, ("autopsy_" + begin_time + "_" + cfg.name))
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            writer = SummaryWriter("")
            eval_nums = 3
            tensorborad_visual_log(net, dataset, writer, 0, eval_nums)
