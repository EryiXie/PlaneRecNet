import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from math import sqrt
import numpy as np

@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter


def bbox_iou(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / union
    return out if use_batch else out.squeeze(0)

def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of size [a, h, w] and [b, h, w].
    The output is of size [a, b].
    """

    masks_a = masks_a.view(masks_a.size(0), -1)
    masks_b = masks_b.view(masks_b.size(0), -1)

    intersection = masks_a.float() @ masks_b.t().float()
    area_a = masks_a.sum(dim=1).unsqueeze(1)
    area_b = masks_b.sum(dim=1).unsqueeze(0)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a


def _scale_size(size, scale):
    """Rescale a size by a ratio.
    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.
    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.
    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
        out (ndarray): The output destination.
    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, dst=out, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imresize_like(img, dst_img, return_scale=False, interpolation='bilinear'):
    """Resize image to the same size of a given image.
    Args:
        img (ndarray): The input image.
        dst_img (ndarray): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.
    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = dst_img.shape[:2]
    return imresize(img, (w, h), return_scale, interpolation)


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.
    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.
    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    """Resize image while keeping the aspect ratio.
    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img

def calc_size_preserve_ar(img_w, img_h, max_size):
    if img_w > img_h:
        w = max_size
        h = img_h / img_w * max_size
    else:
        h = max_size
        w = img_w / img_h * max_size
    return (int(w), int(h))

def pad_even_divided(img, divisor=32):
    h, w, c = img.shape
    ext_h = divisor - h%divisor if h%divisor!=0 else 0 
    ext_w = divisor - w%divisor if w%divisor!=0 else 0 
    padded_img = np.zeros((h+ext_h, w+ext_w, c))
    padded_img[0:h, 0:w, :] = img
    return padded_img


def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()

    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def get_points_coordinate(depth, intrinsic_inv):
    B, C, H, W  = depth.shape
    y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float),
                           torch.arange(0, W, dtype=torch.float)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(H * W), x.view(H * W)
    xyz = torch.stack((x, y, torch.ones_like(x))).cuda()  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(intrinsic_inv, xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, H, W)


def get_surface_normal(point_clouds, valid_condition):
    '''
    Surface normal computation method from GeoNet https://github.com/xjqi/GeoNet
    '''
    B, C, H, W= point_clouds.shape
    point_matrix = F.unfold(point_clouds, kernel_size=5, stride=1, padding=4, dilation=2)
    
    valid_condition = F.unfold(valid_condition, kernel_size=5, stride=1, padding=4, dilation=2)
    valid_condition = valid_condition.view(-1, 1, 5 * 5, H, W)
    valid_condition = valid_condition.permute(0, 3, 4, 2, 1)  # (b, h, w, self.k_size*self.k_size, 1)
    valid_condition_all = valid_condition.repeat([1, 1, 1, 1, 3])
    valid_condition_all = (valid_condition_all > 0.5).cuda()
    
    
    # An = b
    matrix_a = point_matrix.view(B, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
    
    matrix_a_zero = torch.zeros_like(matrix_a)
    matrix_a_valid = torch.where(valid_condition_all, matrix_a, matrix_a_zero)
    matrix_a_trans = matrix_a_valid.transpose(3, 4)
    
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([1, H, W, 25, 1]).float().cuda()

    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a_valid)
    matrix_deter = torch.linalg.det(point_multi.to("cpu")).cuda()
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix.cuda())
    inv_matrix = torch.inverse(inversible_matrix.to("cpu")).cuda()

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2., dim=3)
    return norm_normalize

def PCA_svd(pts):
    mean = pts.mean(dim=0)
    pts_adjust = pts - mean
    H = torch.mm(pts_adjust.transpose(0, 1), pts_adjust)
    eigenvectors, eigenvalues, eigenvectors_T = torch.svd(H)
    return mean, eigenvectors[:,2]


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init