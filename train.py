import os
import time
import argparse
import datetime
import cv2
import numpy as np
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from planerecnet import PlaneRecNet
from data.config import cfg, MEANS, set_cfg, set_dataset
from data.datasets import PlaneAnnoDataset, S2D3DSDataset, ScanNetDataset, detection_collate, enforce_size
from data.augmentations import SSDAugmentation, BaseTransform
from utils.utils import SavePath, MovingAverage
from utils import timer
from models.functions.losses import PlaneRecNetLoss
from models.functions.nms import point_nms

import eval as eval_script

parser = argparse.ArgumentParser(description='PlaneRecNet Training Script')
# Basic Settings
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one.')
parser.add_argument('--config', default='PlaneRecNet_50_config',
                    help='The config object to use.')
parser.add_argument('--save_folder', default='./weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='./logs/',
                    help='Directory for saving logs.')
parser.add_argument('--backbone_folder', default='./weights/',
                    help='Directory for loading Backbone.')                 
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--validation_size', default=2000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--no_tensorboard', dest='no_tensorboard', action='store_true',
                    help='Whether visualize training loss, validation loss and outputs with tensorboard.')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='Automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
parser.add_argument('--reproductablity', dest='reproductablity', action='store_true',
                    help='Set this if you want to reproduct the almost same results as given in the ablation study.')                

# Hyper Parameters for Training
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
# Only related to SGD optimizer
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')

# You might not need customize these
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--save_interval', default=12500, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=10000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))
    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr
print("initial learning step: ", cur_lr)

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['ins', 'lav', 'cat', 'dpt', 'pln']

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    raise AssertionError('Cuda is not avaliable.')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net:PlaneRecNet, criterion:PlaneRecNetLoss):
        super().__init__()
        self.net = net
        self.criterion = criterion
    
    def forward(self, batched_images, batched_gt_instances, batched_gt_depths):
        """
        Args:
            - batched_images: Tensor, each images in (C, H, W) format.
            - batched_gt_instances: Dict of Tensor, ground truth instances.
            - batched_gt_depth: Tensor, ground truth depth map, each in (H, W) format.
        Returns:
            - losses: a dict, losses from PlaneRecNet
        """
        mask_pred, cate_pred, kernel_pred, depth_pred = self.net(batched_images)
        losses = self.criterion(self.net, mask_pred, cate_pred, kernel_pred, depth_pred, batched_gt_instances, batched_gt_depths)
        return losses


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    
    """
    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = self.prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out
    
    @torch.no_grad()
    def prepare_data(self, datum, devices:list=None, allocation:list=None):

        def gradinator(x):
            x.requires_grad = False
            return x
        if devices is None:
            devices = ['cuda:0']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        batched_images, batched_gt_instances, batched_gt_depths = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                batched_images[cur_idx]  = gradinator(batched_images[cur_idx].to(device))
                batched_gt_depths[cur_idx]  = gradinator(batched_gt_depths[cur_idx].to(device))
                for key in batched_gt_instances[cur_idx]:
                    batched_gt_instances[cur_idx][key] = gradinator(batched_gt_instances[cur_idx][key].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = batched_images[random.randint(0, len(batched_images)-1)].size()
            for idx, (image, gt_depth, gt_instances) in enumerate(zip(batched_images, batched_gt_depths, batched_gt_instances)):
                batched_images[idx], batched_gt_depths[idx], batched_gt_instances[idx] \
                    = enforce_size(image, gt_depth, gt_instances, w, h)
        cur_idx = 0
        split_images, split_depths, split_instances = [[None for alloc in allocation] for _ in range(3)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(batched_images[cur_idx:cur_idx+alloc], dim=0)
            split_depths[device_idx]    = torch.stack(batched_gt_depths[cur_idx:cur_idx+alloc], dim=0)
            split_instances[device_idx]   = batched_gt_instances[cur_idx:cur_idx+alloc]
            cur_idx += alloc
        return split_images, split_instances, split_depths


def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = eval(cfg.dataset.name)(image_path=cfg.dataset.train_images, 
                            anno_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))

    setup_eval()
    val_dataset = eval(cfg.dataset.name)(image_path=cfg.dataset.valid_images,
                            anno_file=cfg.dataset.valid_info,
                            transform=BaseTransform(MEANS))

    prn_net = PlaneRecNet(cfg)
    net = prn_net
    net.train()
    
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        prn_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        prn_net.init_weights(backbone_path=args.backbone_folder + cfg.backbone.path)
    
    optimizer = optim.Adam([
        {'params': net.backbone.parameters(), 'lr': 5*args.lr},
        {'params': net.fpn.parameters(), 'lr': args.lr},
        {'params': net.inst_head.parameters(), 'lr': args.lr},
        {'params': net.mask_head.parameters(), 'lr': args.lr},
        {'params': net.depth_decoder.parameters(), 'lr': 2*args.lr}], lr=args.lr)
    
    criterion = PlaneRecNetLoss()

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)
    
    net = CustomDataParallel(NetLoss(net, criterion))
    net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn: prn_net.freeze_bn() # Freeze bn so we don't kill our means
    prn_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: prn_net.freeze_bn(True)

    # Initialize TensorBoardX Writer
    if not args.no_tensorboard:
        begin_time = (datetime.datetime.now()).strftime("%d%m%Y%H%M%S")
        logpath = os.path.join(args.log_folder, (begin_time + "_" + cfg.name))
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        writer = SummaryWriter(logpath)

    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    step_index = 0

    # If Pytorch >= 1.9, please set the generator to utilize cuda to avoid crush.
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True) # Add generator=torch.Generator(device='cuda') for pytorch >= 1.9
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue
            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)
                losses = {k: (v).mean() for k,v in losses.items()} # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 50 == 0:
                    # log losses to tensorboard
                    if not args.no_tensorboard: 
                        log_losses(writer, losses, iteration)
                        if iteration % 5000 == 0 and iteration > 0:
                            log_visual_example(prn_net, val_dataset, writer, iteration)
                    if iteration % 100 == 0:
                        # print losses(moving averaged) to console
                        eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                        total = sum([loss_avgs[k].get_avg() for k in losses])
                        loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                        print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' total: %.3f || ETA: %s || time/batch: %.3fs')
                                % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    prn_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                            if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                                print('Deleting old save...')
                                os.remove(latest)

            # This is done per epoch
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and iteration > 0 and epoch < num_epochs-2:
                # no validation when iteration = 0 or when last epoch
                    compute_validation_metrics(epoch, iteration, prn_net, val_dataset, args.validation_size)
        
        # Compute validation mAP after training is finished
        compute_validation_metrics(epoch, iteration, prn_net, val_dataset)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            prn_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    prn_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr


def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """
    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()


def setup_eval():
    eval_script.parse_args(['--no_bar'])


def compute_validation_metrics(epoch, iteration, prn_net, val_dataset, eval_nums=-1):
    with torch.no_grad():
        prn_net.eval()
        start =  time.time()
        print()
        print("Computing validation metrics (this may take a while)...", flush=True)
        eval_script.evaluate(prn_net,val_dataset, during_training=True, eval_nums=eval_nums)
        end = time.time()
        prn_net.train()


def log_losses(writer: SummaryWriter, losses, iteration):
    """
    Write losses to the event file
    """
    total = 0
    for l, v in losses.items():
        rounded_v = round(v.item(), 5)
        writer.add_scalar("Losses:{}".format(l), rounded_v, iteration)
        total += v
    writer.add_scalar("Losses:{}".format("total"), total, iteration)


def log_visual_example(prn_net: PlaneRecNet, val_dataset: PlaneAnnoDataset, writer: SummaryWriter, iteration, eval_nums=5):
    """
    Write visaul examples to the event file
    """
    with torch.no_grad():
        prn_net.eval()
        start =  time.time()
        eval_script.tensorborad_visual_log(prn_net, val_dataset, writer, iteration, eval_nums)
        end = time.time()
        prn_net.train()


if __name__ == "__main__":
    if args.reproductablity:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('************************Repoductablity Mode**************************')
        print('* Set the random seed for random, np.random, torch and cudnn as {}. *'.format(seed))
        print('************************Repoductablity Mode**************************')

    train()
