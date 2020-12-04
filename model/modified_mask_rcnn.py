import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN, FasterRCNN
import timm
from model.backbone import TimmToVisionFPN, TimmToVision, resnet50_fpn, calculate_param
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model.roi_box_head_extractor import RoIFeatureExtractor, RoIFeatureExtractor_new
from model.roi_box_head_predictor import RoIBoxPredictor
# connect our models here !!


def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # m = timm.create_model('cspresnet50', pretrained=True, num_classes=0, global_pool='')
    # backbone = TimmToVision(m, 1024)
    # backbone = resnet50_fpn()

    m = timm.create_model('cspresnet50', features_only=True, pretrained=True)
    backbone = TimmToVisionFPN(m)

    model = FasterRCNN(backbone, num_classes)
    # model = MaskRCNN(backbone, num_classes)

    model.box_head = RoIFeatureExtractor(num_inputs=256, resolution=7)
    model.box_predictor = RoIBoxPredictor(num_classes)

    return model


if __name__ == "__main__":
    m = get_model(21)
    print(calculate_param(m))
