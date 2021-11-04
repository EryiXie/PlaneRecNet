
"""Partly Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/utils/augmentations.py
    Licensed under The MIT License [see LICENSE for details]
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from numpy import random
from math import sqrt
from data.config import cfg, MEANS, STD

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        for t in self.transforms:
            img, depth, masks, boxes, labels, plane_paras = t(img, depth, masks, boxes, labels, plane_paras)
        return img, depth, masks, boxes, labels, plane_paras


class Resize_and_Pad(object):
    """
    Resize the image to its long side == cfg.max_size, filling the
    area: [(long side - short side) * long side] to with mean and 
    putting the image in the top-left.
    """
    def __init__(self, resize_gt=True, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.pad_gt = pad_gt
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
    
    def __call__(self, image, depth, masks=None, boxes=None, labels=None, plane_paras=None):
        img_h, img_w, channels = image.shape
        
        if img_h != self.max_size or img_w != self.max_size:
            height, width = (self.max_size, int(img_w * (self.max_size / img_h))) if img_h > img_w  else (int(img_h * (self.max_size / img_w)), self.max_size)

            image = cv2.resize(image, (width, height))
            depth = cv2.resize(depth, (width, height))

            if self.resize_gt:
                # Act like each object is a color channel
                masks = masks.transpose((1, 2, 0))
                masks = cv2.resize(masks, (width, height))
                
                # OpenCV resizes a (w,h,1) array to (s,s), so fix that
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, 0)
                else:
                    masks = masks.transpose((2, 0, 1))

                # Scale bounding boxes (which are currently absolute coordinates)
                boxes[:, [0, 2]] *= (width  / img_w)
                boxes[:, [1, 3]] *= (height / img_h)

            expand_image = np.zeros((self.max_size, self.max_size, channels), dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[:height, :width] = image

            expand_depth = np.zeros((self.max_size, self.max_size), dtype=depth.dtype)
            expand_depth[:height, :width] = depth

            if self.pad_gt:
                expand_masks = np.zeros((masks.shape[0], self.max_size, self.max_size), dtype=masks.dtype)
                expand_masks[:,:height,:width] = masks
                masks = expand_masks
        
            # Discard boxes that are smaller than we'd like
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]

            keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
            masks = masks[keep]
            boxes = boxes[keep]
            labels = labels[keep]
            
            return expand_image, expand_depth, masks, boxes, labels, plane_paras
        else:
            # Discard boxes that are smaller than we'd like
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]

            keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
            masks = masks[keep]
            boxes = boxes[keep]
            labels = labels[keep]
            
            return image, depth, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, width, height, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image, depth, masks, boxes=None, labels=None, plane_paras=None):
        im_h, im_w, channels = image.shape

        expand_image = np.zeros(
            (self.height, self.width, channels),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        expand_depth = np.zeros((self.height, self.width), dtype=depth.dtype)
        expand_depth[:im_h, :im_w] = depth

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:,:im_h,:im_w] = masks
            masks = expand_masks

        return expand_image, expand_depth, masks, boxes, labels, plane_paras


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """
    # TODO: change the above line of intro

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size

    def __call__(self, image, depth, masks, boxes, labels, plane_paras):
        img_h, img_w, _ = image.shape

        if img_h != self.max_size and img_w != self.max_size:
            width, height = self.max_size, self.max_size

            image = cv2.resize(image, (width, height))
            depth = cv2.resize(depth, (width, height))

            if self.resize_gt:
                # Act like each object is a color channel
                masks = masks.transpose((1, 2, 0))
                masks = cv2.resize(masks, (width, height))
                
                # OpenCV resizes a (w,h,1) array to (s,s), so fix that
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, 0)
                else:
                    masks = masks.transpose((2, 0, 1))

                # Scale bounding boxes (which are currently absolute coordinates)
                boxes[:, [0, 2]] *= (width  / img_w)
                boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels = labels[keep]

        return image, depth, masks, boxes, labels, plane_paras


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, depth, masks, boxes, labels, plane_paras


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, depth, masks, boxes, labels, plane_paras


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, depth, masks, boxes, labels, plane_paras


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, depth, masks, boxes, labels, plane_paras


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, depth, masks, boxes, labels, plane_paras


class ToCV2Image(object):
    def __call__(self, tensor, depth, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), depth.numpy().astype(np.float32), masks, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, depth, masks=None, boxes=None, labels=None, plane_paras=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), depth, masks, boxes, labels, plane_paras


class RandomMirror(object):
    def __call__(self, image, depth, masks, boxes, labels, plane_paras):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            depth = depth[:, ::-1] # TODO: Is 1 channel the same as 3 channels? 
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            mirror_trans = np.asarray([[-1,0,0],[0,1,0],[0,0,1]])
            plane_paras[:,:3] = np.matmul(mirror_trans, plane_paras[:,:3].transpose()).transpose()
        return image, depth, masks, boxes, labels, plane_paras


class RandomFlip(object):
    def __call__(self, image, depth, masks, boxes, labels, plane_paras):
        height , _ , _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            depth = depth[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
            flip_trans = np.asarray([[1,0,0],[0,-1,0],[0,0,1]])
            plane_paras[:,:3] = np.matmul(flip_trans, plane_paras[:,:3].transpose()).transpose()
        return image, depth, masks, boxes, labels, plane_paras


class RandomRot90(object):
    def __call__(self, image, depth, masks, boxes, labels, plane_paras):
        old_height , old_width , _ = image.shape
        k = random.randint(4)
        image = np.rot90(image,k)
        depth = np.rot90(depth,k)
        masks = np.array([np.rot90(mask,k) for mask in masks])
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2], box[3], old_width - 1 - box[0]] for box in boxes])
            old_width, old_height = old_height, old_width
        rot_90_trans = np.asarray([[0,-1,0],[1,0,0],[0,0,1]])
        plane_paras[:,:3] = np.matmul(rot_90_trans, plane_paras[:,:3].transpose()).transpose()

        return image, depth, masks, boxes, labels, plane_paras
# plane paras should be rotated too


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, depth, masks, boxes, labels, plane_paras):
        im = image.copy()
        im, depth, masks, boxes, labels, plane_paras = self.rand_brightness(im, depth, masks, boxes, labels, plane_paras)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, depth, masks, boxes, labels, plane_paras = distort(im, depth, masks, boxes, labels, plane_paras)
        return im, depth, masks, boxes, labels, plane_paras


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    # TODO: Check how and what to change of this one.
    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img, depth, masks=None, boxes=None, labels=None, plane_paras=None):

        img = img.astype(np.float32)
        depth = depth.astype(np.float32)

        
        if self.transform.normalize:
            img = (img - self.mean) / self.std
            
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]
        

        return img.astype(np.float32), depth.astype(np.float32), masks, boxes, labels, plane_paras


class RandomMotionBlur(object):
    def __init__(self, lower_degree=3, upper_degree=12, angle=180):
        self.upper_degree = upper_degree
        self.lower_degree = lower_degree
        self.angle = angle
        assert self.lower_degree >= 3
        assert self.lower_degree < self.upper_degree
        assert self.angle >= 0

    def __call__(self, image, depth, masks, boxes, labels, plane_paras):

        if random.randint(3) < 1 :
            degree = random.randint(self.lower_degree, self.upper_degree)
            angle = random.randint(0, self.angle)

            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

            motion_blur_kernel = motion_blur_kernel / degree
            blurred = cv2.filter2D(image, -1, motion_blur_kernel)

            # convert to uint8
            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            blurred = np.array(blurred, dtype=np.uint8)

            return blurred, depth, masks, boxes, labels, plane_paras
        else:
            return image, depth, masks, boxes, labels, plane_paras


class RandomGaussianNoise(object):
    def __init__(self, mean=0, var=0.0002):
        self.mean = mean
        self.var = var

    def __call__(self, image, depth, masks, boxes, labels, plane_paras):

        if random.randint(3) < 1:
            image = np.array(image / 255, dtype=float)
            var =  random.randint(5,11)*self.var
            noise = np.random.normal(self.mean, var ** 0.5, image.shape)
            out = image + noise
            if out.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            out = np.clip(out, low_clip, 1.0)
            out = np.uint8(out * 255)
            return out, depth, masks, boxes, labels, plane_paras
        else:
            return image, depth, masks, boxes, labels, plane_paras


###########################################
#  Augmentation/Transformation in Queue  #
##########################################


def do_nothing(img=None, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
    return img, depth, masks, boxes, labels, plane_paras


def enable_if(condition, obj):
    return obj if condition else do_nothing


class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            enable_if(cfg.augment.photometric_distort, PhotometricDistort()),
            enable_if(cfg.augment.random_mirror, RandomMirror()),
            enable_if(cfg.augment.random_flip, RandomFlip()),
            enable_if(cfg.augment.random_rot90, RandomRot90()),
            enable_if(cfg.augment.motion_blur, RandomMotionBlur()),
            enable_if(cfg.augment.gaussian_noise, RandomGaussianNoise()),
            Resize(resize_gt=True),
            #enable_if(not cfg.augment.preserve_aspect_ratio, Pad(cfg.max_size, cfg.max_size, mean)),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        return self.augment(img, depth, masks, boxes, labels, plane_paras)

class BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            Resize(resize_gt=True),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, depth=None, masks=None, boxes=None, labels=None, plane_paras=None):
        return self.augment(img, depth, masks, boxes, labels, plane_paras)


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        img = img.permute(0, 3, 1, 2).contiguous()

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255.
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img
