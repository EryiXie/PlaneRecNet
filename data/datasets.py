import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from data.config import cfg, set_dataset
from pycocotools import mask as maskUtils
import random
import json
import abc

def get_label_map():
    if cfg.dataset.label_map is None:
        return {x+1: x+1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map
        
class PlaneAnnoDataset(data.Dataset):
    """ 
    The general class for reading training and validation datasets. 
    Data sample is organized with a extend format of COCO dataset. https://cocodataset.org/#format-data
    """
    def __init__(self, image_path, anno_file, transform=None,
                 dataset_name=None, has_gt=True, has_pos=True):
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(anno_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.name = dataset_name
        self.has_gt = has_gt
        self.has_pos = has_pos
    
    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, instances, depth).
        '''
        image, instances, depth = self.pull_item(index)
        return image, instances, depth

    def pull_item(self, index):
        '''
        instances: Dict {'masks': [N x H x W], torch.uint8, 
                        'boxes': [N x 4], torch.float64,
                        'classes': [N], torch.int64} in CPU
        N: the number of instance per frame
        '''
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            # Target has {'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path).astype(np.float32)
        height, width, _ = img.shape

        depth_path = self.get_depth_path(file_name)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.has_pos:
            k_matrix = self.get_camera_matrix(file_name)
            s = cfg.dataset.scale_factor
            scale_matrix = np.asarray([[s,0,s],[0,s,s],[0,0,1]])
            k_matrix = scale_matrix * k_matrix
        else:
            k_matrix = np.zeros((0))

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)
            boxes = [list(np.array([obj['bbox'][0], obj['bbox'][1], obj['bbox'][0] + obj['bbox'][2], obj['bbox'][1] + obj['bbox'][3]])) for obj in target]
            # box -> [xmin, ymin, xmax, ymax]
            labels = [get_label_map()[obj['category_id']] - 1  for obj in target]
            boxes = np.array(boxes)
            labels = np.array(labels)
            if cfg.dataset.has_pos:
                plane_paras = self.get_plane_para(target)
                plane_paras = np.array(plane_paras)
            else:
                plane_paras = np.zeros((0))

        if self.transform is not None:
            if len(target) > 0:
                img, depth, masks, boxes, labels, plane_paras = self.transform(img, depth, masks, boxes, labels, plane_paras)
            else:
                img, depth, _, _, _ = self.transform(img, depth, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]))
                masks = None
                boxes = None
                labels = None
                plane_paras = None
        instances = {'masks': torch.from_numpy(masks), 'boxes': torch.from_numpy(boxes), 'classes': torch.from_numpy(labels), 'plane_paras': torch.from_numpy(plane_paras), 'k_matrix': torch.from_numpy(k_matrix)}

        target = np.array(target)
        if target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))
        
        return torch.from_numpy(img).permute(2, 0, 1), instances, torch.from_numpy(depth).unsqueeze(dim=0) * cfg.dataset.depth_resolution
    
    def __len__(self):
        return len(self.ids)
    

    def pull_image(self, index):
        '''Returns the original image object at index in OPENCV form (BGR)
        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)
    
    def pull_depth(self, index):
        '''Returns the original depth map object at index in uint16
        Argument:
            index (int): index of img to show
        Return:
            numpy ndarray dtpye: uint16
        '''
        img_id = self.ids[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        dep_path = self.get_depth_path(img_path)
        return cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)


    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @abc.abstractmethod
    def get_depth_path(self, rgb_file_name):
        return
    @abc.abstractmethod
    def get_camera_matrix(self, rgb_file_name):
        return
    @abc.abstractmethod
    def get_plane_para(self, target):
        return


class ScanNetDataset(PlaneAnnoDataset):

    def __init__(self, image_path, anno_file, transform=None,
                 dataset_name=None, has_gt=True, has_pos=True):
        super(ScanNetDataset, self).__init__(image_path, anno_file, transform, dataset_name, has_gt, has_pos)
     
    def get_depth_path(self, rgb_file_name):
        depth_file_name = rgb_file_name.replace("color", "depth").replace(".jpg", ".png")
        depth_path = osp.join(self.root, depth_file_name)
        return depth_path

    def get_camera_matrix(self, rgb_file_name):
        sens_name = rgb_file_name.split('/')[0]
        pose_file_name = os.path.join(sens_name, "frame", "intrinsic", sens_name + ".txt")
        pose_path = os.path.join(self.root, pose_file_name)
            
        f = open(pose_path)
        lines = f.readlines()
        f.close()
        words = lines[9].split(' ')
        k_matrix = np.asarray([float(words[i]) for i in range(2,18)]).reshape((4,4))[:3,:3]
        return k_matrix

    def get_plane_para(self, target):
        #planeOffsets = [np.array([obj['plane_paras'][3]]) for obj in target]
        planes = [list(np.array([obj['plane_paras'][0], obj['plane_paras'][1], obj['plane_paras'][2], obj['plane_paras'][3]])) for obj in target]
        return planes


class NYUDataset(PlaneAnnoDataset):

    def __init__(self, image_path, anno_file, transform=None,
                 dataset_name=None, has_gt=True, has_pos=True):
        super(NYUDataset, self).__init__(image_path, anno_file, transform, dataset_name, has_gt, has_pos)
    
    def get_depth_path(self, rgb_file_name):
        depth_root = self.root.replace("images", "depths")
        depth_file_name = rgb_file_name.replace(".jpg", ".png")
        depth_path = osp.join(depth_root, depth_file_name)
        return depth_path


class S2D3DSDataset(PlaneAnnoDataset):

    def __init__(self, image_path, anno_file, transform=None,
                 dataset_name=None, has_gt=True, has_pos=True):
        super(S2D3DSDataset, self).__init__(image_path, anno_file, transform, dataset_name, has_gt, has_pos)
    
    def get_depth_path(self, rgb_file_name):
        depth_root = self.root.replace("images", "depths")
        depth_file_name = rgb_file_name.replace("rgb", "depth").replace(".jpg", ".png")
        depth_path = osp.join(depth_root, depth_file_name)
        print(depth_path)
        return depth_path
    
    def get_camera_matrix(self, rgb_file_name):
        pose_root = self.root.replace('images_val', 'poses').replace('images', 'poses')
        pose_file_name = rgb_file_name.replace('rgb', 'pose').replace('.jpg', '.json')
        pose_path = os.path.join(pose_root, pose_file_name)
        f = open(pose_path)
        pose = json.load(f)
        f.close()
        k_matrix = np.asarray(pose['camera_k_matrix'])
        return k_matrix

    def get_plane_para(self, target):
        return [list(np.array([obj['plane_paras'][0], obj['plane_paras'][1], obj['plane_paras'][2], obj['plane_paras'][3], obj['plane_paras'][4], obj['plane_paras'][5]])) for obj in target]


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images, depth maps and list of instances(dict)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) {'bboxes': boxes, 'masks': masks, 'classes': labels} instance annotations for a given image are stacked
                on 0 dim. The output gt is a list of dicts.
            3) (tensor) batch of depth maps stacked on their 0 dim
    """
    imgs = []
    depths = []
    instances = []

    for sample in batch:
        imgs.append(sample[0])
        instances.append(sample[1])
        depths.append(sample[2])

    return imgs, instances, depths


def enforce_size(img, depth, instances, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, depth, instances
        
        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        depth = F.interpolate(depth.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        depth.squeeze_(0)

        # Act like each object is a color channel
        instances['masks'] = F.interpolate(instances['masks'].unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        instances['masks'].squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        instances['boxes'][:, [0, 2]] *= (w_prime / new_w)
        instances['boxes'][:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img   = F.pad(  img, pad_dims, mode='constant', value=0)
        depth = F.pad(depth, pad_dims, mode='constant', value=0)
        instances['masks'] = F.pad(instances['masks'], pad_dims, mode='constant', value=0)

        return img, depth, instances


# Just for testing
if __name__ == "__main__":
    import argparse
    from data.config import cfg, set_cfg, MEANS
    from data.augmentations import SSDAugmentation
    from models.functions.funcs import get_points_coordinate

    def parse_args(argv=None):
        parser = argparse.ArgumentParser(description="Debbuging datasets.")
        parser.add_argument(
            "--dataset",
            default=None,
            type=str,
            help='The dataset config object to use',
        )
        parser.add_argument(
            "--config", 
            default="PlaneRecNet_50_config", 
            help="The network config object to use.")
        global args
        args = parser.parse_args(argv)
    

    parse_args()

    set_cfg(args.config)
    set_dataset(args.dataset)
    print(cfg.backbone.name, cfg.backbone.path)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    
    dataset = ScanNetDataset(image_path=cfg.dataset.valid_images, 
                            anno_file=cfg.dataset.valid_info,
                            transform=SSDAugmentation(MEANS))
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                  num_workers=2,
                                  shuffle=False,
                                  collate_fn=detection_collate)

    for idx, data_ele in enumerate(data_loader):
        imgs, gt_instances, gt_depths = data_ele
        intrinsic_matrix = torch.stack([gt_instances[img_idx]['k_matrix'] for img_idx in range(len(gt_instances))], dim=0).cuda()
        gt_depths = torch.stack([gt_depths[img_idx] for img_idx in range(len(gt_depths))], dim=0)
        intrinsic_inv = torch.inverse(intrinsic_matrix).float().to("cuda") # (B, 4, 4)
        point_clouds = get_points_coordinate(gt_depths.permute(0,2,3,1).to("cuda"), instrinsic_inv=intrinsic_inv)

        for i in range(len(imgs)):
            inst = gt_instances[i]
            masks = inst['masks']
            offsets = inst['plane_paras'][:,3:].cuda().double()
            normals = inst['plane_paras'][:,:3].cuda().double()
            total = 0
            print("gt masks: {}, gt planes: {}".format(masks.shape, inst['plane_paras'].shape))
            error = 0
            for j in range(masks.shape[0]):
                pts = point_clouds[i][:, masks[j]]
                valid_mask = (pts[2,:] > 0)
                pts = pts[:, valid_mask].double()
                offset = offsets[j]
                normal = normals[j]
                dist = torch.abs(torch.matmul(pts.permute(1,0), normal) - offset).to("cuda").double()
                error = error + dist.mean()

            print(error.item() / masks.shape[0])
            print()
        print()
        if idx >= 5000:
            break
