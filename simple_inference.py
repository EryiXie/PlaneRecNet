"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/eval.py
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import cv2
import os
from pathlib import Path
from numpy.core.numeric import NaN
import torch
from torch.nn.functional import interpolate
from planerecnet import PlaneRecNet
from data.augmentations import FastBaseTransform
from data.config import set_cfg, cfg, COLORS
from utils import timer
from models.functions.funcs import PCA_svd
from models.functions.funcs import calc_size_preserve_ar, pad_even_divided
from collections import defaultdict
import numpy as np
import scipy.io

color_cache = defaultdict(lambda: {})

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PlaneRecNet Inference")
    parser.add_argument("--trained_model",default=None, type=str, help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument("--config", default="PlaneRecNet_50_config", help="The config object to use.")
    # Inference Settings
    parser.add_argument("--image", default=None, type=str, help='Inference with a single image.')
    parser.add_argument("--images", default=None, type=str, help='Inference with multiple images.')
    parser.add_argument("--max_img", default=0, type=int, help="The maximum number of inference images in a folder.")
    parser.add_argument("--ibims1", default=None, type=str, help="Only for iBims-1 outputs")
    parser.add_argument("--ibims1_pd", default=None, type=str, help="test plane depth")
    # Display Args (Default: mask, bbox, score and class label display are enabled.)
    parser.add_argument("--no_mask", action="store_true", help="Whether to draw object masks or not.")
    parser.add_argument("--no_box", action="store_true", help="Whether to draw object bounding boxes or not.")
    parser.add_argument("--no_text", action="store_true", help="Whether to draw object scores and categories or not.")
    # Inference Parameters
    parser.add_argument('--top_k', default=100, type=int, help='Further restrict the number of predictions to parse')
    parser.add_argument("--nms_mode", default="matrix", type=str, choices=["matrix", "mask"], help='Choose NMS type from matrix and mask nms.')
    parser.add_argument('--score_threshold', default=0.3, type=float, help='Detections with a score under this threshold will not be considered.')
    parser.add_argument("--depth_mode", default="colored", type=str, choices=["colored", "gray"], help='Choose visualization mode of depth map')
    parser.add_argument('--depth_shift', default=512, type=float, help='Depth shift')
    global args
    args = parser.parse_args(argv)


def display_on_frame(result, frame, mask_alpha=0.5, fps_str='', no_mask=False, no_box=False, no_text=False):
    frame_gpu = frame / 255.0
    h, w, _ = frame.shape

    pred_scores = result["pred_scores"]
    pred_depth = result["pred_depth"].squeeze()
    
    if pred_scores is None:
        return frame.byte().cpu().numpy(), pred_depth.cpu().numpy()
    
    pred_masks = result["pred_masks"].unsqueeze(-1)
    pred_boxes = result["pred_boxes"]
    pred_classes = result["pred_classes"]
    num_dets = pred_scores.size()[0]

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color


    if not no_mask and num_dets>0:
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=frame_gpu.device.index).view(
            1, 1, 1, 3) for j in range(num_dets)], dim=0)
        masks_color = pred_masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = pred_masks * (-mask_alpha) + 1
        for j in range(num_dets):
            frame_gpu = frame_gpu * inv_alph_masks[j] + masks_color[j]
            

        frame_numpy = (frame_gpu * 255).byte().cpu().numpy()
        for j in range(num_dets):
            masks_color_np = pred_masks[j].cpu().squeeze().numpy().astype(np.uint8)
            contours, hierarchy = cv2.findContours(masks_color_np, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_numpy,contours,-1,(255,255,255),1)

        if not no_text or not no_box:
            for j in reversed(range(num_dets)):
                x1, y1, x2, y2 = pred_boxes[j].int().cpu().numpy()
                color = get_color(j)
                score = pred_scores[j].detach().cpu().numpy()

                if not no_box:
                    cv2.rectangle(frame_numpy, (x1, y1), (x2, y2), color, 1)
                
                if not no_text:
                    _class = cfg.dataset.class_names[pred_classes[j].cpu().numpy()]
                    text_str = '%s: %.2f' % (_class, score)

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (x1, y1 + text_h + 1)
                    text_color = [255, 255, 255]

                    cv2.rectangle(frame_numpy, (x1, y1),(x1 + text_w, y1 + text_h + 4), color, -1)
                    cv2.putText(frame_numpy, text_str, text_pt, font_face,font_scale, text_color, font_thickness, cv2.LINE_AA)
        if not no_text:
            score = pred_scores[j].detach().cpu().numpy()
            _class = cfg.dataset.class_names[pred_classes[j].cpu().numpy()]
            text_str = '%s: %.2f' % (_class, score)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(
                text_str, font_face, font_scale, font_thickness)[0]
            text_pt = (x1, y1 + text_h + 1)
            text_color = [255, 255, 255]

            cv2.rectangle(frame_numpy, (x1, y1),
                        (x1 + text_w, y1 + text_h + 4), color, -1)
            cv2.putText(frame_numpy, text_str, text_pt, font_face,
                        font_scale, text_color, font_thickness, cv2.LINE_AA)
                
        return frame_numpy, pred_depth.cpu().numpy()
    else:
        return frame.byte().cpu().numpy(), pred_depth.cpu().numpy()


def inference_image(net: PlaneRecNet, path: str, save_path: str = None, depth_mode: str='colored'):
    frame_np = cv2.imread(path)
    H, W, _ = frame_np.shape

    if frame_np is None:
        return
    frame_np = cv2.resize(frame_np, calc_size_preserve_ar(W, H, cfg.max_size), interpolation=cv2.INTER_LINEAR)
    frame_np = pad_even_divided(frame_np) #pad image to be evenly divided by 32

    frame = torch.from_numpy(frame_np).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    results = net(batch)

    blended_frame, depth = display_on_frame(results[0], frame, no_mask=args.no_mask, no_box=args.no_box, no_text=args.no_text)

    if save_path is None:
        name, ext = os.path.splitext(path)
        save_path = name + '_seg' + ext
        depth_path = name + '_dep.png'
    else:
        name, ext = os.path.splitext(save_path)
        depth_path = name + '_dep.png'
        
    cv2.imwrite(save_path, blended_frame)

    if depth_mode == 'colored':
        vmin = np.percentile(depth, 1)
        vmax = np.percentile(depth, 99)
        depth = depth.clip(min=vmin, max=vmax)
        depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(depth_path, depth_color)
    elif depth_mode == 'gray':
        depth = (depth*args.depth_shift).astype(np.uint16)
        cv2.imwrite(depth_path, depth)
    
   
def inference_images(net: PlaneRecNet, in_folder: str, out_folder: str, max_img: int=0, depth_mode: str='colored'):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print()
    index = 0
    input_list = list(Path(in_folder).glob('*'))
    max_img = min(max_img, len(input_list)) if max_img > 0 else len(input_list)
    for p in sorted(input_list):
        img_path = str(p)
        name, ext = os.path.splitext(os.path.basename(img_path))
        if ext != ".png" and ext != ".jpg":
            continue
        out_path = os.path.join(out_folder, name+ext)
        inference_image(net, img_path, out_path, depth_mode=depth_mode)
        print("Inference images: " + os.path.basename(img_path) + ' -> ' + os.path.basename(out_path),  end='\r')
        index = index + 1
        if index >= max_img:
            break
    print()
    print("Done.")


def ibims1(net: PlaneRecNet, in_folder: str, out_folder: str):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print()
    index = 0
    input_list = list(Path(in_folder).glob('*'))
    for p in sorted(input_list):
        img_path = str(p)
        name, ext = os.path.splitext(os.path.basename(img_path))
        depth_out_path = os.path.join(out_folder, name+"_results.mat")
        if ext != ".mat":
            continue
        out_path = os.path.join(out_folder, name+ext)
        image_data = scipy.io.loadmat(img_path)  
        data = image_data['data']
        # extract neccessary data
        rgb = data['rgb'][0][0]   # RGB image
        if rgb is None:
            return
        frame = torch.from_numpy(rgb).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        results = net(batch)
        pred_depth = results[0]["pred_depth"].squeeze().cpu().numpy()
        scipy.io.savemat(depth_out_path, {'pred_depths': pred_depth})

        vmin = np.percentile(pred_depth, 1)
        vmax = np.percentile(pred_depth, 99)
        pred_depth = pred_depth.clip(min=vmin, max=vmax)
        pred_depth = ((pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(pred_depth, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(depth_out_path.replace('.mat','.png'), depth_color)
        print(os.path.basename(img_path) + ' -> ' + os.path.basename(out_path),  end='\r')
        index = index + 1
        
    print()
    print("Done.")


def ibims1_pd(net: PlaneRecNet, in_folder: str, out_folder: str):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print()
    index = 0
    input_list = list(Path(in_folder).glob('*'))
    for p in sorted(input_list):
        img_path = str(p)
        name, ext = os.path.splitext(os.path.basename(img_path))
        depth_out_path = os.path.join(out_folder, name+"_results.mat")
        if ext != ".mat":
            continue
        out_path = os.path.join(out_folder, name+ext)
        image_data = scipy.io.loadmat(img_path)  
        data = image_data['data']
        calib = data['calib'][0][0]
        # extract neccessary data
        rgb = data['rgb'][0][0]   # RGB image
        if rgb is None:
            return
        frame = torch.from_numpy(rgb).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        results = net(batch)
        pred_depth = results[0]["pred_depth"]#.squeeze().cpu().numpy()
        pred_masks = results[0]["pred_masks"]
        if pred_masks is not None:

            k_matrix = calib.transpose()
            k_matrix = torch.from_numpy(k_matrix).double().cuda()
            intrinsic_inv = torch.inverse(k_matrix).double().cuda()

            B, C, H, W  = pred_depth.shape

            cx = k_matrix[0][2]
            cy = k_matrix[1][2]
            fx = k_matrix[0][0]
            fy = k_matrix[1][1]
            # convert to point clouds
            v, u = torch.meshgrid(torch.arange(H), torch.arange(W))
            Z = pred_depth.squeeze(dim=0)
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud = torch.cat((X,Y,Z), dim=0).permute(1,2,0)

            N = pred_masks.shape[0]
            plane_depths = []
            x = torch.arange(W, dtype=torch.float32).view(1, W).repeat(H, 1)
            y = torch.arange(H, dtype=torch.float32).view(H, 1).repeat(1, W)
            xy1 = torch.stack((x, y, torch.ones((H, W)))).view(3, -1).double()
            k_inv_dot_xy1 = torch.matmul(intrinsic_inv.squeeze(), xy1)
            for idx in range(0,N):
                mask = pred_masks[idx].bool()
                point_cloud_seg = point_cloud[mask, :].squeeze(dim=0)
                center, normal = PCA_svd(point_cloud_seg)
                plane_depths.append(torch.dot(center, normal) / torch.matmul(normal, k_inv_dot_xy1))
            

            plane_depths = torch.stack(plane_depths, dim=0)
            plane_depths = plane_depths.view(-1, H, W)
            pred_depth = pred_depth.squeeze()
            
            for i in range(plane_depths.shape[0]):
                pred_depth = torch.where(pred_masks[i], plane_depths[i].float(), pred_depth)
        else:
            pred_depth = pred_depth.squeeze()

        pred_depth = pred_depth.cpu().numpy()
        pred_depth[pred_depth<=0] = NaN
        pred_depth[pred_depth>=10] = NaN

        scipy.io.savemat(depth_out_path, {'pred_depths': pred_depth})
        
        vmin = np.percentile(pred_depth, 1)
        vmax = np.percentile(pred_depth, 99)
        pred_depth = pred_depth.clip(min=vmin, max=vmax)

        pred_depth = ((pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(pred_depth, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(depth_out_path.replace('.mat','.png'), depth_color)
        
        print(os.path.basename(img_path) + ' -> ' + os.path.basename(out_path),  end='\r')
        index = index + 1
        
    print()
    print("Done.")


if __name__ == "__main__":
    nms_config = parse_args()
    timer.disable_all()
    new_nms_config = {
        'nms_type': args.nms_mode, 
        'mask_thr': args.score_threshold, 
        'update_thr': args.score_threshold,
        'top_k': args.top_k,}

    set_cfg(args.config)

    cfg.solov2.replace(new_nms_config)
    #cfg.solov2.print()

    net = PlaneRecNet(cfg)
    if args.trained_model is not None:
        net.load_weights(args.trained_model)
    else:
        net.init_weights(backbone_path="weights/" + cfg.backbone.path)
        print(cfg.backbone.name)
        
    net.train(mode=False)
    net = net.cuda()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            print('Inference image: {}'.format(inp))
            inference_image(net, inp, out, depth_mode=args.depth_mode)
        else:
            print('Inference image: {}'.format(args.image))
            inference_image(net, args.image, depth_mode=args.depth_mode)
    
    if args.images is not None:
        inp, out = args.images.split(':')
        inference_images(net, inp, out, max_img=args.max_img, depth_mode=args.depth_mode)
    if args.ibims1 is not None:
        inp, out = args.ibims1.split(':')
        ibims1(net, inp, out)
    if args.ibims1_pd is not None:
        inp, out = args.ibims1_pd.split(':')
        ibims1_pd(net, inp, out)