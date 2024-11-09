# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget,FasterRCNNBoxScoreTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="D:\理工姿态识别\VIT\deep-high-resolution-net.pytorch-master\experiments\coco\hrnet\w32_256x192_adam_lr1e-3_ASFF_BIBlock4.yaml",
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
   # ################可视化#################################################################################
   #  import cv2 as cv
   #  import numpy as np
   #  from PIL import Image
   #  from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
   #  from PIL import Image
   #  import matplotlib.pyplot as plt
   #  def visulize_spatial_attention(img_path, attention_mask, ratio=1, cmap="jet"):
   #      """
   #      img_path:   image file path to load
   #      save_path:  image file path to save
   #      attention_mask: 2-D attention map with np.array type, e.g, (h, w) or (w, h)
   #      ratio:  scaling factor to scale the output h and w
   #      cmap:   attention style, default: "jet"
   #      """
   #      print("load image from: ", img_path)
   #      img = Image.open(img_path, mode='r')
   #      img_h, img_w = img.size[0], img.size[1]
   #      plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
   #
   #      # scale the image
   #      # scale表示放大或者缩小图片的比率
   #      img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
   #      img = img.resize((img_h, img_w))
   #      plt.imshow(img, alpha=1)
   #      plt.axis('off')
   #
   #      mask = cv.resize(attention_mask, (img_h, img_w))
   #      normed_mask = mask / mask.max()
   #      normed_mask = (normed_mask * 255).astype('uint8')
   #      plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
   #      plt.show()
   #
   #  target_layers = [model.module.biblock_stages[0][0]]
   #  # img = cv.imread("D:\coco2017\images\\val2017\\000000000785.jpg")
   #  # image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
   #  image = np.array(Image.open("D:\coco2017\images\\val2017\\000000000785.jpg"))
   #  image = cv.resize(image, (192, 256))
   #  rgb_img = np.float32(image) / 255
   #  input_tensor = preprocess_image(rgb_img,
   #                              mean=[0.485, 0.456, 0.406],
   #                              std=[0.229, 0.224, 0.225])
   #  cam = EigenCAM(model,
   #                 target_layers,
   #                 )
   #  car_category = 1
   #  ##########HOOK##############
   #
   #  features_in_hook = []
   #  features_out_hook = []
   #  # 使用 hook 函数
   #  def hook(module, fea_in, fea_out):
   #      features_in_hook.append(fea_in)         # 勾的是指定层的输入
   #      # 只取前向传播的数值
   #      features_out_hook.append(fea_out.data)      # 勾的是指定层的输出
   #      return None
   #
   #  layer_name = 'module.biblock_stages.0.1.drop_path'
   #  for (name, module) in model.named_modules():
   #      print(name)
   #      if name == layer_name:
   #          module.register_forward_hook(hook=hook)
   #  #########################################################
   #  model(input_tensor)
   #  amp=np.array(features_out_hook[0].mean(3)[0:].to("cpu"))[0]
   #  visulize_spatial_attention("D:\coco2017\images\\val2017\\000000000785.jpg",amp)
   #  from visualize import inspect_atten_map_by_locations
   #
   #  inspect_atten_map_by_locations(img, model, query_locations, model_name="transposer", mode='dependency', save_img=True, threshold=0.0)
   #  targets = [FasterRCNNBoxScoreTarget(labels="labels", bounding_boxes=None)]
   #  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
   #  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
   #
   #  # In this example grayscale_cam has only one image in the batch:
   #  grayscale_cam = grayscale_cam[0, :]
   #  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
   #  plt.imshow(visualization)
   #  plt.show()
   #  # You can also get the model outputs without having to re-inference
   #  model_outputs = cam.outputs
   #
   #  ################可视化#################################################################################

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
