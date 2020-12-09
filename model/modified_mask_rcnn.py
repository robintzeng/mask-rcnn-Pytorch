import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection import roi_heads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN, FasterRCNN

import timm
from model.backbone import TimmToVisionFPN, TimmToVision, resnet50_fpn, calculate_param
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .cascade_rpn_head import CascadeRPNHead
from torchvision.ops import MultiScaleRoIAlign

from model.roi_box_head_extractor import RoIFeatureExtractor, RoIFeatureExtractor_new
from model.roi_box_head_predictor import RoIBoxPredictor
# connect our models here !!

from .IA_faster_rcnn import FasterRCNNIA
# connect our models here !!
import pdb

def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # m = timm.create_model('cspresnet50', pretrained=True, num_classes=0, global_pool='')
    # backbone = TimmToVision(m)
    # m = timm.create_model('cspresnet50', features_only=True, pretrained=True)
    m = timm.create_model('cspresnet50', features_only=True, pretrained=True, pretrained_strict=False)
    backbone = TimmToVisionFPN(m)
    # m = timm.create_model('cspresnet50', pretrained=True, num_classes=0, global_pool='')
    # backbone = TimmToVision(m,1024)
    #backbone = resnet50_fpn()
    # model = MaskRCNN(backbone, num_classes)
    # backbone = resnet50_fpn()
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)


    # ["0"] rather than [0]
    out_channels = backbone.out_channels
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    rpn_head = CascadeRPNHead(out_channels, feat_channels=out_channels, num_anchors=num_anchors, stage=2)
    
    # model = FasterRCNN(backbone, num_classes=num_classes, rpn_head=rpn_head)
    model = FasterRCNN(backbone, num_classes=num_classes)

    # IA branch
    # model = FasterRCNNIA(backbone, num_classes=num_classes, rpn_head=rpn_head)

    # Box head
    model.roi_heads.box_head = RoIFeatureExtractor(num_inputs=256, resolution=7)
    model.roi_heads.box_predictor = RoIBoxPredictor(num_classes)

    return model


if __name__ == "__main__":
    m = get_model(21)
    print(calculate_param(m))
