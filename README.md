# PlaneRecNet
This is an official implementation for PlaneRecNet: A multi-task convolutional neural network provides instance segmentation for piece-wise planes and monocular depth estimation, and focus on the cross-task consistency between two branches.
![Network Architecture](/data/network_architecture.png)
## Changing Logs
22th. Oct. 2021: Initial update, some trained models and data annotation will be uploaded very soon.

29th. Oct. 2021: Upload ResNet-50 based model.

3rd. Nov. 2021: Nice to know that "prn" or "PRN" is a forbidden name in Windows.

4th. Nov. 2021: For inference, input image will be resized to max(H, W) == cfg.max_size, and reserve the aspect ratio. Update enviroment.yml, so that newest GPU can run it as well.

24th.Jan.2022: Add "auto padding" function to simple_inference.py, fixed config invalid in eval.py, **fixed misimplemented mAP metric** (very sorry about that, how can I be that careless?) and update results table. Upload evaluation annotation samples of ScanNet dataset. (Next update, I will fix pathing issue and distributed data parallel issue.)
# Installation
## Install environment:
- Clone this repository and enter it:
```Shell
git clone https://github.com/EryiXie/PlaneRecNet.git
cd PlaneRecNet
```

- Set up the environment using one of the following methods:
  - Using [Anaconda](https://www.anaconda.com/distribution/)
    - Run `conda env create -f environment.yml `
  - Using [Docker](https://www.docker.com/get-started)
    - *` dockerfile will come later...`*

## Download trained model:
Here are our models (released on Oct 22th, 2021), which can reproduce the results in the [paper](https://arxiv.org/abs/2110.11219):

![Quantitative Results](/data/prn_results_table.png)

All models below are trained with batch_size=8 and a single RTX3090 or a single RTXA6000 on the [plane annotation](https://github.com/NVlabs/planercnn) for [ScanNet](http://www.scan-net.org/) dataset:

|  Image Size  | Backbone|  FPS  |  Weights |
| ------ | --------------- | --- |  - |
|480x640   | Resnet50-DCN | 19.1 | [PlaneRecNet_50](https://drive.google.com/file/d/1DdT6eEd7Ba-eUshs1PtPpn7WUtLeTyd9/view?usp=sharing)
|480x640 |   Resnet101-DCN    | 14.4 | [PlaneRecNet_101](https://drive.google.com/file/d/1rDnYjBGD-yMO4dzuwtTCs2s53QkEpa9l/view?usp=sharing)|



# Simple Inference
Inference with an single image(*.jpg or *.png format):
```Shell
python3 simple_inference.py --config=PlaneRecNet_101_config --trained_model=weights/PlaneRecNet_101_9_125000.pth  --image=data/example_nyu.jpg
```

Inference with images in a folder:
```Shell
python3 simple_inference.py --config=PlaneRecNet_101_config --trained_model=weights/PlaneRecNet_101_9_125000.pth --images=input_folder:output_folder
```

Inference with .mat files from [iBims-1](https://www.asg.ed.tum.de/lmf/ibims1/) Dataset:
```Shell
python3 simple_inference.py --config=PlaneRecNet_101_config --trained_model=weights/PlaneRecNet_101_9_125000.pth --ibims1=input_folder:output_folder
```

Then you will get segmentation and depth estimation results like these:

![Qualititative Results](/data/prn_results_vis.png)


# Training
PlaneRecNet is trained on ScanNet with 100k samples on one single RTX 3090 with batch_size=8, it takes approximate **37 hours**. Here are the [data annotations](https://drive.google.com/file/d/1HMaJ_gaAoP6s_KNf1jNydQgOA_5cukaL/view?usp=sharing)(about 1.0 GB) for training, validation and evaluation on ScanNet datasets, which is based on the annotation given by [PlaneRCNN](https://github.com/NVlabs/planercnn) and converted into json file. Newly uploaded evaluation samples are refined with the rejection threshold mentioned the paper of PlaneRCNN, it results some difference, but very limited. *Please not that, our training sample is not the same as PlaneRCNN, because we don't have their training split at hand.*

**Please notice, the pathing and naming rules in our data/dataset.py, is not compatable with the raw data extracted with the ScanNetv2 original code. Please refer to [this issue](https://github.com/EryiXie/PlaneRecNet/issues/2) for fixing tips, thanks [uyoung-jeong](https://github.com/uyoung-jeong) for that.** I will add the data preprocessing script to fix this, once I have time.

Of course, please download [ScanNet](http://www.scan-net.org/) too for rgb image, depth image and camera intrinsic etc.. The annotation file we provide only contains paths for images and camera intrinsic and the ground truth of piece-wise plane instance and its plane parameters.

- To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
 - Run one of the training commands below.
   - Press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.


Trains PlaneRecNet_101_config with a batch_size of 8.
```Shell
python3 train.py --config=PlaneRecNet_101_config --batch_size=8
```
 Trains PlaneRecNet, without writing any logs to tensorboard.
 ```Shell
python3 train.py --config=PlaneRecNet_101_config --batch_size=8 --no_tensorboard
```
Run Tensorboard on local dir "./logs" to check the visualization. So far we provide loss recording and image sample visualization, may consider to add more (22.Oct.2021).
```Shell
tenosrborad --logdir /log/folder/
``` 
Resume training PlaneRecNet with a specific weight file and start from the iteration specified in the weight file's name.
```Shell
python3 train.py --config=PlaneRecNet_101_config --resume=weights/PlaneRecNet_101_X_XXXXX.pth
```

Use the help option to see a description of all available command line arguments.
```Shell
python3 train.py --help
```


## Multi-GPU Support

**###Multi-GPU Mode is not working, I will fix it in the next update, and switch to DDP!###**

We adapted the Multi-GPU support from [YOLACT](https://github.com/dbolya/yolact), as well as the introduction of how to use it as follow:

 - Put `CUDA_VISIBLE_DEVICES=[gpus]` on the beginning of the training command.
   - Where you should replace [gpus] with a comma separated list of the index of each GPU you want to use (e.g., 0,1,2,3).
   - You should still do this if only using 1 GPU.
   - You can check the indices of your GPUs with `nvidia-smi`.
 - Then, simply set the batch size to `8*num_gpus` with the training commands above. The training script will automatically scale the hyperparameters to the right values.
   - If you have memory to spare you can increase the batch size further, but keep it a multiple of the number of GPUs you're using.
   - If you want to allocate the images per GPU specific for different GPUs, you can use `--batch_alloc=[alloc]` where [alloc] is a comma seprated list containing the number of images on each GPU. This must sum to `batch_size`.


# Known Issues

1. Userwarning of torch.max_pool2d. **This has no real affect**. It appears when using PyTorch 1.9. And it is claimed "[fixed](https://github.com/pytorch/pytorch/issues/60053)" for the nightly version of PyTorch. 
```Shell
UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
```

2. Userwarning of leaking Caffe2 while training. This [issues](https://github.com/pytorch/pytorch/issues/57273) related to dataloader in PyTorch1.9, to avoid showing this warning, set **pin_memory=False** for dataloader. **But you don't necessarily need to do this**.
```Shell
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
```

3. Fixed missimplemented mAP metric. I modified the mAP metric code at the very early stage of the impelementation, and because here we only detect one single class, I made the code a little bit easiler and I made a mistake:

```
def compute_segmentation_metrics(ap_data, gt_masks, gt_boxes, gt_classes, pred_masks, pred_boxes, pred_classes, pred_scores):
     ...
     # THAT THE LINE THAT COMPELETELY WRONG, which used to be: num_gt_for_class = 1
     # num_gt_for_class is not "numbers of classes in gt", it is NUMBERS OF GT INSTANCES OF ONE SINGLE CLASS IN ONE INPUT IMAGE!
     num_gt_for_class = sum([1 for x in gt_classes if x == 0]) 
     ...
```
As described, the old wrongly implemented and misunderstood "num_gt_for_class=1" resulted that the average percision was calculated only considering one instance per image. Therefore, the results were completely **wrong**. But luckily, I use the same code to evaluate PlaneRCNN and PlaneAE, so the conclusion of the paper is still valid, in my opinion.


# Citation
If you use PlaneRecNet or this code base in your work, please cite:
```
@misc{xie2021planerecnet,
      title={PlaneRecNet: Multi-Task Learning with Cross-Task Consistency for Piece-Wise Plane Detection and Reconstruction from a Single RGB Image}, 
      author={Yaxu Xie and Fangwen Shu and Jason Rambach and Alain Pagani and Didier Stricker},
      year={2021},
      eprint={2110.11219},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Contact
For questions about our paper or code, please contact [Yaxu Xie](mailto:yaxu.xie@dfki.de), or take a good use at the [**Issues**](https://github.com/EryiXie/PlaneRecNet/issues) section of this repository.
