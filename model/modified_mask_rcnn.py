import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection import roi_heads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN, FasterRCNN
import timm
from model.backbone import TimmToVisionFPN, TimmToVision, resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .IA_mask_rcnn import MaskRCNNIA
# connect our models here !!


def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    # Without FPN cannot be used with MaskRCNNIA because of the version of the torchvision
    m = timm.create_model('cspresnet50', pretrained=True, num_classes=0, global_pool='')
    backbone = TimmToVision(m,1024)
    
    ## with FPN 
    #m = timm.create_model('cspresnet50', features_only=True, pretrained=True)
    #backbone = TimmToVisionFPN(m)
    
    ## FPN official version
    #backbone = resnet50_fpn()
    
    model = MaskRCNNIA(backbone, num_classes)

    return model

