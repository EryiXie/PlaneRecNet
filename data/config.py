"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/data/config.py
    Licensed under The MIT License [see LICENSE for details]
"""

from models.backbone import ResNetBackbone
import torch

COLORS = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139),
)


# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

PLANE_CLASSES = ('plane',)
PLANE_LABEL_MAP = {1: 1}

# ----------------------- CONFIG CLASS ----------------------- #


class Config(object):
    '''
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    '''

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        '''
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        '''

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        '''
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        '''
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

# ----------------------- DATASETS ----------------------- #

dataset_base = Config({
    'name': 'PlaneAnnoDataset',

    # Training images and annotations
    'train_images': '',
    'train_info':   '',

    # Validation images and annotations.
    'valid_images': '',
    'valid_info':   '',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,
    'has_pos': True,

    # A list of names for each of you classes.
    'class_names': PLANE_CLASSES,
    'label_map':  PLANE_LABEL_MAP,

    # The ratio to convert depth pixel value to meter
    'depth_resolution': None,
    'min_depth':  None,
    'max_depth':  None,
    # Resize scale factor
    'scale_factor':  None,
})

scannet_dataset = dataset_base.copy({
    'name': 'ScanNetDataset',

    # Training images and annotations
    'train_images': './scannet/scans/',
    'train_info':   './scannet/scannet_train.json',

    # Validation images and annotations.
    'valid_images': './scannet/scans/',
    'valid_info':   './scannet/scannet_val.json',

    # Evaluation images and annotations.
    'eval_images': './scannet/scans/',
    'eval_info':   './scannet/scannet_eval.json',

    # A list of names for each of you classes.
    'class_names': PLANE_CLASSES,
    'label_map':  PLANE_LABEL_MAP,

    # The ratio to convert depth pixel value to meter
    'depth_resolution': 1/1000,
    'min_depth': 1/1000,
    'max_depth': 40,
    # Scale factor to resize the camera intrinsic matrix for project predicted depth to point cloud
    'scale_factor': 1,
})

nyu_eval = dataset_base.copy({
    'name': 'NYUDataset',

    # Evaluation images and annotations.
    'eval_images': './NYU/nyu_images/',
    'eval_info': './NYU/nyu_eval.json',
    # Resize scale factor
    'scale_factor': 1,
    'min_depth': 1/1000,
    'max_depth': 40,
    'has_pos': False,
    'depth_resolution': 1 / 65535.0 * 9.99547
})

S2D3DS_dataset = dataset_base.copy({
# Training on Stanford 2D-3D-S dataset may also gives you some nice results, since the depth ground truth has high quality.
# But I labeled the piece-wise plane segmentation a very long times ago as a student, and I lost the back-traceability because I very uncarefully renamed all images.
# So I would like not to make the annotation files available, I am so sorry for that.
    'name': 'S2D3DSDataset',

    # Training images and annotations
    'train_images': './S2D3DS/images/',
    'train_info':   './S2D3DS/s2d3ds_train.json',

    # Validation images and annotations.
    'valid_images': './S2D3DS/images_val/',
    'valid_info':   './S2D3DS/s2d3ds_val.json',

    # The ratio to convert depth pixel value to meter
    'depth_resolution': 1/512,
    'min_depth': 1/512,
    'max_depth': 40,
    # Resize scale factor
    'scale_factor': 0.5,
})

# ----------------------- DATA AUGMENTATION ---------------- #

data_augment = Config(
    {
    # Randomize hue, vibrance, etc.
    'photometric_distort': True,
    # Mirror the image with a probability of 1/2
    'random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'random_flip': True,
    # With uniform probability, rotate the image [0,90,180,270] degrees, if the input image is not square, you need some modification.
    'random_rot90': False,
    # With mothin blur, no need if you train on ScanNet, ScanNet is already blurrrrrred...
    'motion_blur': False,
    # With Gaussian nosie
    'gaussian_noise': False,
    }
)

# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config(
    {
        'channel_order': 'RGB',
        'normalize': True,
        'subtract_means': False,
        'to_float': False,
    }
)


# ----------------------- BACKBONES ----------------------- #

backbone_base = Config(
    {
        'name': 'Base Backbone',
        'path': 'path/to/pretrained/weights',
        'type': object,
        'args': tuple(),
        'transform': resnet_transform,
        'selected_layers': list(),
    }
)

resnet101_backbone = backbone_base.copy(
    {
        'name': 'ResNet101',
        'path': 'resnet101_reducedfc.pth',
        'type': ResNetBackbone,
        'args': ([3, 4, 23, 3],),
        'transform': resnet_transform,
        'selected_layers': list(range(3, 7)),
    }
)

resnet101_dcn_inter3_backbone = resnet101_backbone.copy(
    {
        'name': 'ResNet101_DCN_Interval3', 'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
    }
)

resnet50_backbone = resnet101_backbone.copy(
    {
        'name': 'ResNet50',
        'path': 'resnet50-19c8e357.pth',
        'type': ResNetBackbone,
        'args': ([3, 4, 6, 3],),
        'transform': resnet_transform,
    }
)

resnet50_dcnv2_backbone = resnet50_backbone.copy(
    {
        'name': 'ResNet50_DCNv2', 'args': ([3, 4, 6, 3], [0, 4, 6, 3]),
    }
)

# ----------------------- Feature Pyramid Network DEFAULTS ----------- #

fpn_base = Config(
    {
        # The selected ResNet output layers as input for FPN
        'selected_layers': list(range(0, 4)),
        # The start layer 0 -> c2
        'start_level': None,
        # The number of features to have in each FPN layer
        'num_features': 256,
        # The upsampling mode used
        'interpolation_mode': 'bilinear',
        # High level mode to use 'retina' or 'orginal' or None
        'high_level_mode' : None,
        # Whether to add relu to the regular layers
        'relu_pred_layers': True,
    }
)

# ----------------------- Depth DECODER DEFAULTS ------------------- #

depth_fpn = Config(
    {
        # The selected ResNet output layers as input for depth decoder
        'selected_layers': list(range(0, 4)),
        # The selected skip connection layers
        'skip_layers': list(range(0, 4)),
        # Whether use ReflectionPad2d
        'use_refle': True,
    }
)

# ----------------------- SOLOV2 DECODER DEFAULTS ------------------- #

solov2_base = Config(
    {
        ### Maks Head Settings
        # The number of prediction kernels
        'num_kernels': 256,
        # Masks input features
        'masks_in_features': ['p2', 'p3', 'p4', 'p5'],
        # The numbers of masks channels
        'masks_channels': 128,

        ### Instance Head Settings
        # The number of prediction masks
        'num_masks': 256,
        # Instance input features
        'instance_in_features': ['p2', 'p3', 'p4', 'p5', 'p6'],
        # The numbers of instance channels
        'instance_channels': 512, # which should can be decrease?
        # FPN instance strides
        'fpn_instance_strides': [8, 8, 16, 32, 32],
        # FPN scale ranges, as the author of solo said 
        # the setting inherited from https://github.com/taokong/FoveaBox
        'fpn_scale_ranges': ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        # The numbers of grids
        'num_grids': [40, 36, 24, 16, 12],
        # Instance Head Depth
        'num_instance_convs': 4,
        # Whether use deformable convolution in instance head
        'use_dcn_in_instance': False,
        # Define the instance center location box with 0.5 * width(or heigth) * sigma
        'sigma': 0.2,
        
        ### Matrix NMS Settings
        # Maximum number of priors before NMS
        'nms_pre': 500,
        # Score Threshold before NMS
        'score_thr': 0.1,
        # NMS Type:
        'nms_type': "matrix",
        # Mask Threshold for Mask NMS
        'mask_thr': 0.1,
        # Update Threshold for Matrix NMS
        'update_thr': 0.15,
        # Matrix NMS kernel type: gaussian OR linear.
        'nms_kernel': 'gaussian',
        # Sigma
        'nms_sigma': 2,
        # Maximum number of output instance per image
        'top_k': 100,

        ### Other Settings
        # Whether use coord conv
        'use_coord_conv': True,
        # Type of norm method to use
        'norm' : 'GN',
        # Pi bias for stable focal loss at beginning 
        'focal_loss_init_pi': 0.01,
    }
)

solov2_light = Config(
# We use the light version of solov2, it turned out to be better than the base setting. I did the research, trust me.
    {
        ### Maks Head Settings
        # The number of prediction kernels
        'num_kernels': 128,
        # Masks input features
        'masks_in_features': ['p2', 'p3', 'p4', 'p5'],
        # The numbers of masks channels
        'masks_channels': 128,

        ### Instance Head Settings
        # The number of prediction masks
        'num_masks': 128,
        # Instance input features
        'instance_in_features': ['p2', 'p3', 'p4', 'p5'],
        # The numbers of instance channels
        'instance_channels': 256, # which should can be decrease?
        # FPN instance strides
        'fpn_instance_strides': [8, 8, 16, 32],
        # FPN scale ranges, as the author of solo said 
        # the setting inherited from https://github.com/taokong/FoveaBox
        'fpn_scale_ranges': ((1, 128), (64, 256), (128, 512), (256, 2048)),
        # The numbers of grids
        'num_grids': [40, 36, 24, 16],
        # Instance Head Depth
        'num_instance_convs': 3,
        # Whether use deformable convolution in instance head
        'use_dcn_in_instance': False,
        # Define the instance center location box with 0.5 * width(or heigth) * sigma
        'sigma': 0.2,
        
        ### Matrix NMS Settings
        # Maximum number of priors before NMS
        'nms_pre': 500,
        # Score Threshold before NMS
        'score_thr': 0.1,
        # NMS Type:
        'nms_type': "matrix",
        # Mask Threshold for Mask NMS
        'mask_thr': 0.1,
        # Update Threshold for Matrix NMS
        'update_thr': 0.15,
        # Matrix NMS kernel type: gaussian OR linear.
        'nms_kernel': 'gaussian',
        # Sigma
        'nms_sigma': 2,
        # Maximum number of output instance per image
        'top_k': 100,

        ### Other Settings
        # Whether use coord conv
        'use_coord_conv': True,
        # Type of norm method to use
        'norm' : 'GN',
        # Pi bias for stable focal loss at beginning 
        'focal_loss_init_pi': 0.01,
    }
)

# ----------------------- PlaneRecNet CONFIGS ----------------------- #

PlaneRecNet_base_config = Config(
    {
        'name': 'PlaneRecNet_base',

         # Dataset Settings
        'dataset': scannet_dataset,
        'num_classes': len(scannet_dataset.class_names) + 1,

        # Data Augmentations
        'augment': data_augment,
        
        # Training Settings
        'max_iter': 125000,
        'lr_steps': (62500, 100000),
        # dw' = momentum * dw - lr * (grad + decay * w)
        'lr': 1e-4,
        'momentum': 0.9,
        'decay': 5e-4,

        'freeze_bn': False,
        # Warm Up Learning Rate
        'lr_warmup_init': 1e-6,
        'lr_warmup_until': 2000,
        # For each lr step, what to multiply the lr with
        'gamma': 0.1,

        # A list of settings to apply after the specified iteration. Each element of the list should look like
        # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
        'delayed_settings': [],

        # Backbone Settings
        'backbone': resnet101_backbone.copy(
            {
                'selected_layers': list(range(2, 4)),
            }
        ),

        # FPN Settings
        'fpn': fpn_base.copy(
            {
                'start_level': 0,
                'high_level_mode' : 'original',
            }
        ),

        # Depth Decoder Settings
        'depth': depth_fpn,

        # SoloV2  Settings
        'solov2': solov2_base,

        # Loss Settings
        'dice_weight': 3.0,
        'focal_weight': 1.0,
        'depth_weight': 5.0,
        'use_lava_loss': False,
        'use_plane_loss': False,

        'lava_weight': 0.5,
        'pln_weight': 1.0,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,

        # Discard detections with width and height smaller than this (in absolute width and height)
        'discard_box_width': 4 / 640,
        'discard_box_height': 4 / 640,

        # Image Size
        'max_size': 640,
        # Device
        'device': 'cuda',
        # Whether or not to preserve aspect ratio when resizing the image.
        # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
        # If False, all images are resized to max_size x max_size
        'preserve_aspect_ratio': False,
    }
)

PlaneRecNet_101_config = PlaneRecNet_base_config.copy(
    {
        'name': 'PlaneRecNet_101',
        'lr_steps': (62500, 100000),
        # Backbone Settings
        'backbone': resnet101_dcn_inter3_backbone.copy(
            {
                'selected_layers': list(range(2, 4)),
            }
        ),

        # FPN Settings
        'fpn': fpn_base.copy(
            {
                'start_level': 0,
                'high_level_mode' : None,
            }
        ),

        # SOLOV2_base  Settings
        'solov2': solov2_light.copy({
            'instance_in_features': ['p2', 'p3', 'p4', 'p5'],
            'num_grids': [40, 36, 24, 16],
            'fpn_instance_strides': [8, 8, 16, 32],
        }),

        'use_lava_loss': True,
        'use_plane_loss': True,
        'lava_weight': 1.0,
        'pln_weight': 1.0,
    }
)

PlaneRecNet_50_config = PlaneRecNet_101_config.copy(
    {
        'name': 'PlaneRecNet_50',
        # Backbone Settings
        'backbone': resnet50_dcnv2_backbone.copy(
            {
                'selected_layers': list(range(2, 4)),
            }
        ),
    }
)

# Default config
cfg = PlaneRecNet_base_config.copy()

def set_cfg(config_name: str):
    ''' Sets the active config. Works even if cfg is already imported! '''
    global cfg

    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]


def set_dataset(dataset_name: str):
    ''' Sets the dataset of the current config. '''
    cfg.dataset = eval(dataset_name)


# Just for Testing
if __name__ == "__main__" :
    set_cfg('PlaneRecNet_50_config')
    cfg.print()

