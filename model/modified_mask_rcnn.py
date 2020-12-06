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
from model.backbone import TimmToVisionFPN, TimmToVision, resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .cascade_rpn_head import CascadeRPNHead
# connect our models here !!
import pdb

def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # m = timm.create_model('cspresnet50', pretrained=True, num_classes=0, global_pool='')
    # backbone = TimmToVision(m)
    # m = timm.create_model('cspresnet50', features_only=True, pretrained=True)
    m = timm.create_model('ECAcspresnet50', features_only=True, pretrained=True, pretrained_strict=False)
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
    
    model = FasterRCNN(backbone,
                       num_classes=num_classes, rpn_head=rpn_head)

    # model = MaskRCNN(backbone,
    #                  num_classes=num_classes,
    #                  rpn_anchor_generator=anchor_generator,
    #                  rpn_head=rpn_head)

    # get the number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
    #                                                    num_classes)

    return model
