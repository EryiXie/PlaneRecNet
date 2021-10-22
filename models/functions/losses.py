import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from data.config import cfg
from models.functions.funcs import imrescale, center_of_mass
from models.functions.vnl import VNL_Loss


class PlaneRecNetLoss(nn.Module):
    """ Joint Weighted Loss Function for PlaneRecNet
    Compute Targets:
        1) Classification loss
        2) Segmentation loss
        3) Depth estimation loss
        4) Depth gradient loss for enhance segmentation
        5) Planar surface normal loss for enhance depth estimation
    """

    def __init__(self):
        super(PlaneRecNetLoss, self).__init__()
        
        self.use_lava_loss = True
        
        # Paras for post-process
        self.num_classes = cfg.num_classes
        self.instance_strides = cfg.solov2.fpn_instance_strides
        self.num_grids = cfg.solov2.num_grids
        self.scale_ranges = cfg.solov2.fpn_scale_ranges
        self.strides = cfg.solov2.fpn_instance_strides
        self.sigma = cfg.solov2.sigma

        # Paras for losses
        self.focal_loss_alpha = cfg.focal_alpha
        self.focal_loss_gamma = cfg.focal_gamma
        self.ins_loss_weight = cfg.dice_weight
        self.conf_loss_weight = cfg.focal_weight
        self.depth_loss_weight = cfg.depth_weight
        self.lava_loss_weight = cfg.lava_weight
        self.pln_loss_weight = cfg.pln_weight

        self.depth_resolution = cfg.dataset.depth_resolution
        self.dataset_name = cfg.dataset.name

        # Losses funcs
        self.inst_loss = DiceLoss()
        self.conf_loss = SigmoidFocalLoss(gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha, reduction="sum")
        self.point_wise_depth_loss = RMSElogLoss(reduction="mean")
        self.depth_constraint_inst_loss = LavaLoss()
        self.vnl = VNL_Loss((480,640))
        

    def forward(self, net, mask_preds, cate_preds, kernel_preds, depth_preds, gt_instances, gt_depths):
        """

        """ 

        losses = {}

        mask_feat_size = mask_preds.size()[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for img_idx in range(len(gt_instances)):
            cur_ins_label_list, cur_cate_label_list, cur_ins_ind_label_list, cur_grid_order_list = self.prepare_ground_truth(gt_instances[img_idx], mask_feat_size=mask_feat_size)
            ins_label_list.append(cur_ins_label_list)
            cate_label_list.append(cur_cate_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
        
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred_list = []
        ins_pred_batched_list = [torch.empty(0)]*len(mask_preds) 
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = mask_preds[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
                ins_pred_batched_list[idx] = torch.cat((ins_pred_batched_list[idx], cur_ins_pred), dim=0)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)
        
        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # Instance Segmentation Loss
        loss_ins = [] # Yaxu: len(ins_pred_list) = tower levels
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(self.inst_loss(input, target))
        loss_ins_mean = torch.cat(loss_ins).mean()
        loss_ins = loss_ins_mean * self.ins_loss_weight
        losses['ins'] = loss_ins


        # Classification Loss
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        pos_inds = torch.nonzero(flatten_cate_labels != self.num_classes).squeeze(1)
        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1
        loss_cate = self.conf_loss_weight * self.conf_loss(flatten_cate_preds, flatten_cate_labels_oh) / (num_ins + 1)
        losses['cat'] = loss_cate


        # Point-wise Depth Loss
        gt_depths = Variable(gt_depths, requires_grad=False)
        depth_preds = F.interpolate(depth_preds, scale_factor=2, mode='bilinear', align_corners=False)
        valid_mask = (gt_depths > cfg.dataset.min_depth) # All ground truth >= min depth are considered as invalid/non-informative pixels
        gt_depths.clamp(max=cfg.dataset.max_depth)
        loss_depth = self.depth_loss_weight * self.point_wise_depth_loss(depth_preds, gt_depths, valid_mask)
        losses['dpt'] = loss_depth

        
        # Plane Surface Normal Constraint Depth Estimation Loss
        if cfg.use_plane_loss:
            loss_plane = []
            B = len(gt_instances)
            intrinsic_matrix = torch.stack([gt_instances[img_idx]['k_matrix'] for img_idx in range(len(gt_instances))], dim=0)
            for img_idx in range(0, B):
                gt_masks = gt_instances[img_idx]['masks'].bool()
                gt_planes = gt_instances[img_idx]['plane_paras']
                gt_depth = gt_depths[img_idx]
                gt_plane_normals = gt_planes[:, :3]
                gt_plane_offsets = gt_planes[:, 3]
                k_matrix = intrinsic_matrix[img_idx]
                loss_plane_per_frame = self.vnl(depth_preds[img_idx], gt_masks, gt_plane_normals, gt_depth, k_matrix)
                loss_plane.append(loss_plane_per_frame)
            loss_plane_mean = torch.stack(loss_plane).mean()
            losses['pln'] = loss_plane_mean * self.pln_loss_weight
            
            
        # Depth Gradient Constraint Instance Segmentation Loss
        if cfg.use_lava_loss:
            loss_lava = []
            valid_mask = None
            if self.dataset_name == 'ScanNet':
                valid_mask = torch.zeros_like(gt_depths)
                valid_mask[:, :, 20:460, 20:620] = 1
            if self.dataset_name == 'Stanford 2D3DS':
                # dilate the valid mask, to filter out invalid gradient values
                valid_mask = gt_depths>0
                dilate_kernel = torch.autograd.Variable(torch.ones((1, 1, 5, 5)).cuda()) 
                invalid_mask = valid_mask.logical_not().float()
                dilate_valid_mask = F.conv2d(invalid_mask, dilate_kernel, padding=2).bool().logical_not()
                valid_mask = dilate_valid_mask
            #batched_gt_scale_invariant_gradients = (compute_gradient_map(depth_preds, valid_mask) / torch.pow(depth_preds.clamp(min=self.depth_resolution), 2)).detach()
            batched_gt_scale_invariant_gradients = (compute_gradient_map(gt_depths, valid_mask) / torch.pow(gt_depths.clamp(min=self.depth_resolution), 2))
            batched_gt_scale_invariant_gradients = batched_gt_scale_invariant_gradients.clamp(max=1e-2)
            batched_gt_scale_invariant_gradients[batched_gt_scale_invariant_gradients<1e-4] = 0

            for idx, ins_pred_per_img in enumerate(ins_pred_batched_list):
                ins_mask_num = ins_pred_per_img.shape[0]
                ins_pred_heat_maps = ins_pred_per_img.sigmoid()
                if ins_mask_num > 0 and batched_gt_scale_invariant_gradients[idx].sum() > 0:
                    loss_lava.append(self.depth_constraint_inst_loss(ins_pred_heat_maps, batched_gt_scale_invariant_gradients[idx]))
            if len(loss_lava) != 0:
                loss_lava_mean = torch.stack(loss_lava).mean()
                loss_lava = loss_lava_mean * self.lava_loss_weight
            else:
                loss_lava = torch.tensor([0.])
            losses['lav'] = loss_lava
        return losses

    @torch.no_grad()
    def prepare_ground_truth(self, gt_instances_per_frame, mask_feat_size):
        # ins_label_list: masks label (mask map size) list in tower levels
        # cate_label_list: class label map (grids size) list in tower levels
        # ins_ind_label_list: bool map (grids size) in tower levels
        # grid_order_list: in tower levels
        gt_bboxes_raw = gt_instances_per_frame['boxes']
        gt_labels_raw = gt_instances_per_frame['classes']
        gt_masks_raw = gt_instances_per_frame['masks']
        device = gt_labels_raw[0].device

        # ins: gt_bboxes -> [xmin, ymin, xmax, ymax]s
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for (lower_bound, upper_bound), stride, num_grid in zip(self.scale_ranges, self.strides, self.num_grids):
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, self.num_classes) 
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices] 
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma 
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            center_ws, center_hs = center_of_mass(gt_masks) 
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0 

            output_stride = 4
            gt_masks = gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            gt_masks = imrescale(gt_masks, scale=1./output_stride)
            if len(gt_masks.shape) == 2:
                gt_masks = gt_masks[..., None]
            gt_masks = torch.from_numpy(gt_masks).to(dtype=torch.uint8, device=device).permute(2, 0, 1)

            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

class LavaLoss(nn.Module):
    '''
    Depth gradient Loss for instance segmentation
    '''

    def __init__(self):
        super(LavaLoss, self).__init__()
        pass
    
    def forward(self, seg_masks, gradient_map):
        gt_size = gradient_map.shape[1:]
        seg_masks = F.interpolate(seg_masks.unsqueeze(0), size=gt_size, mode='bilinear').squeeze(0)
        lava_loss_per_img = seg_masks.mul(gradient_map)
        loss = lava_loss_per_img.sum() / (gradient_map.sum() * seg_masks.shape[0])
        return loss

@torch.no_grad()
def compute_gradient_map(depth_map, valid_mask=None):
    '''
    Compute gradient map from depth map with 3x3 sobel filter
    '''
    sobel_x = torch.Tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    sobel_x = sobel_x.view((1, 1, 3, 3))
    sobel_x = torch.autograd.Variable(sobel_x.cuda())

    sobel_y = torch.Tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
    sobel_y = sobel_y.view((1, 1, 3, 3))
    sobel_y = torch.autograd.Variable(sobel_y.cuda())
    
    depth_map_padded = F.pad(depth_map, pad=(1,1,1,1), mode='reflect') # Don't use zero padding mode, you know why.
    gx = F.conv2d(depth_map_padded, (1.0 / 8.0) * sobel_x, padding=0)
    gy = F.conv2d(depth_map_padded, (1.0 / 8.0) * sobel_y, padding=0)
    gradients = torch.pow(gx, 2) + torch.pow(gy, 2)

    if valid_mask is not None:
        gradients = gradients * valid_mask
    
    return gradients

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "none"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        pass
    
    def forward(self, input, target):
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d


class RMSElogLoss(nn.Module):
    def __init__(self, clamp_val=1e-9, reduction: str = "none"):
        super(RMSElogLoss, self).__init__()
        self.clamp_val = clamp_val
        self.reduction = reduction
    
    def forward(self, input, target, valid_mask):
        N, C, H, W = input.shape
        input = input.view(N, C*H*W)
        target = target.view(N, C*H*W)
        valid_mask = valid_mask.view(N, C*H*W)

        l1 = torch.abs(torch.log(input.clamp(min=self.clamp_val)) - torch.log(target.clamp(min=self.clamp_val))).mul(valid_mask)
        batched_mean = torch.sum(l1 ** 2, dim=1) / torch.sum(valid_mask, dim=1)
        batched_loss = torch.sqrt(batched_mean)

        if self.reduction == "mean":
            loss = batched_loss.mean()
        elif self.reduction == "sum":
            loss = batched_loss.sum()

        return loss


