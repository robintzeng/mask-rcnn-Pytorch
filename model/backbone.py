from torchvision.models.detection import MaskRCNN, FasterRCNN
import torch
import timm
import torchvision
import torchvision.models as models
import torchvision.models.detection.backbone_utils as backbone_utils
from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class TimmToVisionFPN(nn.Module):
    def __init__(self, backbone):
        super(TimmToVisionFPN, self).__init__()
        self.backbone = backbone
        self.out_channels = 256
        ## if you set timm model = resnet rather than cspresnet, you should set 
        ## in_channels_list = [256, 512, 1024, 2048]
        self.in_channels_list = [128, 256, 512, 1024]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x):
        x = self.backbone(x)
        out = OrderedDict()
        for i in range(len(x)-1):
            out[str(i)] = x[i+1]
        out = self.fpn(out)
        return out


class TimmToVision(nn.Module):
    def __init__(self, backbone, out_channels):
        super(TimmToVision, self).__init__()
        self.backbone = backbone
        self.out_channels = out_channels

    def forward(self, x):
        x = self.backbone(x)
        return x


def resnet50_fpn():
    backbone = backbone_utils.resnet_fpn_backbone('resnet50', True)
    return backbone


def calculate_param(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params



