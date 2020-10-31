import torch
import timm
import torchvision
import torchvision.models as models
import torchvision.models.detection.backbone_utils as backbone_utils
from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# No yet workable


class TimmToVisionFPN(nn.Module):
    def __init__(self, backbone):
        super(TimmToVisionFPN, self).__init__()
        self.backbone = backbone
        self.out_channels = 256
        self.in_channels_list = [64, 256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x):
        x = self.backbone(x)
        out = OrderedDict([(k, v) for k, v in zip(range(len(x)), x)])
        out = self.fpn(out)
        return out


class TimmToVision(nn.Module):
    def __init__(self, backbone):
        super(TimmToVision, self).__init__()
        self.backbone = backbone
        self.out_channels = 1024

    def forward(self, x):
        x = self.backbone(x)
        return x


def test():
    input = torch.Tensor(4, 3, 640, 640)

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    # resnet = torchvision.models.resnet50(pretrained=False)
    # resnet = torchvision.models._utils.IntermediateLayerGetter(resnet, return_layers)
    # resnet_out = resnet(input)
    # for k, v in resnet_out.items():
    #     print(k, v.shape)

    m = timm.create_model('cspresnet50', pretrained=True, num_classes=91, global_pool='')
    print(m(input).shape)
    # m = torchvision.models.mobilenet_v2(pretrained=True).features
    # print(m(input).shape)

    # m = TimmToVisionFPN(m)
    # o = m(input)

    # for (k, v) in o.items():
    #     print(k, v.shape)

    # backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=4)
    # o = backbone(input)
    # for (k, v) in o.items():
    #     print(k, v.shape)
    backbone = models.resnet50()
    # return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    # in_channels_list = [256, 512, 1024, 2048]
    # out_channels = 256
    # resnet_with_fpn = backbone_utils.BackboneWithFPN(backbone,
    #                                                  return_layers, in_channels_list, out_channels)

    # output = resnet_with_fpn(input)
    # print(type(output))
    # for (k, v) in output.items():
    #     print(k)
    #     print(v.shape)

    # resnet = torchvision.models.resnet50(pretrained=False)
    # in_channels_stage2 = resnet.inplanes // 8
    # print(in_channels_stage2)
    # in_channels_list = [
    #     in_channels_stage2,
    #     in_channels_stage2 * 2,
    #     in_channels_stage2 * 4,
    #     in_channels_stage2 * 8,
    # ]
    # print(in_channels_list)
    # out_channels = 256
    # fpn = FeaturePyramidNetwork(
    #     in_channels_list=in_channels_list,
    #     out_channels=out_channels,
    #     extra_blocks=LastLevelMaxPool(),
    # )
    # fpn_out = fpn(resnet_out)
    # for k, v in fpn_out.items():
    #     print(k, v.shape)
