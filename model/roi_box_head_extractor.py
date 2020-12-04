import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.resnet import BasicBlock, Bottleneck

from .non_local import NONLocalBlock2D_Group
from .non_local import ListModule


class FPNUpChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNUpChannels, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # top
        self.top = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # bottom
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # top
        out = self.top(x)
        # bottom
        out0 = self.bottom(x)
        # residual
        out1 = out + out0
        out1 = self.relu(out1)

        return out1


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def make_fc(dim_in, hidden_dim, use_gn=False):
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


class RoIFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, num_inputs=1280, resolution=7):
        super(RoIFeatureExtractor, self).__init__()

        input_size = num_inputs * resolution ** 2
        representation_size = 1024

        nonlocal_use_bn = True
        nonlocal_use_relu = True
        nonlocal_use_softmax = False
        nonlocal_use_ffconv = True
        nonlocal_use_attention = False
        nonlocal_inter_channels = 512

        # add conv and pool like faster rcnn
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        out_channels = 1024

        self.nonlocal_conv = FPNUpChannels(num_inputs, out_channels)

        # shared non-local
        shared_num_group = 4
        self.shared_num_stack = 1
        shared_nonlocal = []
        for i in range(self.shared_num_stack):
            shared_nonlocal.append(
                NONLocalBlock2D_Group(out_channels, num_group=shared_num_group, inter_channels=nonlocal_inter_channels,
                                      sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu,
                                      use_softmax=nonlocal_use_softmax, use_ffconv=nonlocal_use_ffconv,
                                      use_attention=nonlocal_use_attention))
        self.shared_nonlocal = ListModule(*shared_nonlocal)

        # seperate group non-local, before fc6 and fc7
        cls_num_group = 4
        self.cls_num_stack = 0

        reg_num_group = 4
        self.reg_num_stack = 1

        nonlocal_use_bn = True
        nonlocal_use_relu = True

        cls_nonlocal = []
        for i in range(self.cls_num_stack):
            cls_nonlocal.append(
                NONLocalBlock2D_Group(out_channels, num_group=cls_num_group, inter_channels=nonlocal_inter_channels,
                                      sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu,
                                      use_softmax=nonlocal_use_softmax, use_ffconv=nonlocal_use_ffconv,
                                      use_attention=nonlocal_use_attention))
        self.cls_nonlocal = ListModule(*cls_nonlocal)

        reg_nonlocal = []
        for i in range(self.reg_num_stack):
            reg_nonlocal.append(
                NONLocalBlock2D_Group(out_channels, num_group=reg_num_group, inter_channels=nonlocal_inter_channels,
                                      sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu,
                                      use_softmax=nonlocal_use_softmax, use_ffconv=nonlocal_use_ffconv,
                                      use_attention=nonlocal_use_attention))
        self.reg_nonlocal = ListModule(*reg_nonlocal)

        # mlp
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.fc7 = make_fc(representation_size, representation_size, use_gn=False)

    def forward(self, x):
        x_conv = x

        identity = x  # [N, 1280, 7, 7]
        x_conv = self.nonlocal_conv(x_conv)
        # shared
        for i in range(self.shared_num_stack):
            x_conv = self.shared_nonlocal[i](x_conv)

        # seperate
        # x_cls = x_conv
        x_reg = x_conv
        # for i in range(self.cls_num_stack):
        #     x_cls = self.cls_nonlocal[i](x_cls)
        for i in range(self.reg_num_stack):
            x_reg = self.reg_nonlocal[i](x_reg)

        # x_cls = self.avgpool(x_cls)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        x_reg = self.avgpool(x_reg)
        x_reg = x_reg.view(x_reg.size(0), -1)

        # MLP
        identity = identity.view(identity.size(0), -1)

        identity = F.relu(self.fc6(identity))
        identity = F.relu(self.fc7(identity))

        return tuple((x_reg, identity))


class RoIFeatureExtractor_new(nn.Module):
    def __init__(self, in_features, num_classes, pretrained=False):
        super(RoIFeatureExtractor_new, self).__init__()
        self.fc_head = TwoMLPHead(in_channels=1280*7*7, representation_size=in_features)
        layers = [
            BasicBlock(256*5, 1024*5), Bottleneck(1024*5, 1024*5),
            BasicBlock(256*5, 1024*5), Bottleneck(1024*5, 1024*5),
            BasicBlock(256*5, 1024*5), Bottleneck(1024*5, 1024*5)
        ]
        self.conv_head = nn.Sequential(*layers)
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

    def forward(self, features):  # N, 1280, 7, 7
        print(features.shape)
        fc_feature = self.fc_head.forward(features)
        conv_feature = self.conv_head(features)
        avgPool = nn.AvgPool2d((conv_feature.shape[2], conv_feature.shape[3]))
        conv_feature = avgPool(conv_feature)

        return (fc_feature, conv_feature)


'''
**********************************************************************************
'''


# class PlainBlock(nn.Module):
#   def __init__(self, Cin, Cout, downsample=False):
#     super().__init__()
#     self.net = nn.Sequential(
#       nn.BatchNorm2d(Cin),
#       nn.ReLU(),
#       nn.Conv2d(Cin, Cout, 3, stride=1+int(downsample), padding=1),
#       nn.BatchNorm2d(Cout),
#       nn.ReLU(),
#       nn.Conv2d(Cout, Cout, 1, padding=1)
#     )

#   def forward(self, x):
#     return self.net(x)


# class ResidualBlock(nn.Module):
#   def __init__(self, Cin, Cout, downsample=False):
#     super().__init__()
#     self.block = PlainBlock(Cin, Cout, downsample)
#     if downsample:
#       self.shortcut = nn.Conv2d(Cin, Cout, 1, stride=2)
#     else:
#       self.shortcut = nn.Identity() if Cin == Cout else nn.Conv2d(Cin, Cout, 1)

#   def forward(self, x):
#     return self.block(x) + self.shortcut(x)


# class ResidualBottleneckBlock(nn.Module):
#   def __init__(self, Cin, Cout, downsample=False):
#     super().__init__()
#     self.block = nn.Sequential(
#       nn.BatchNorm2d(Cin),
#       nn.ReLU(),
#       nn.Conv2d(Cin, Cout // 4, 1),
#       nn.BatchNorm2d(Cout // 4),
#       nn.ReLU(),
#       nn.Conv2d(Cout // 4, Cout // 4, 3, padding=1),
#       nn.BatchNorm2d(Cout // 4),
#       nn.ReLU(),
#       nn.Conv2d(Cout // 4, Cout, 1)
#     )
#     self.shortcut = nn.Identity() if Cin == Cout else nn.Conv2d(Cin, Cout, 1)

#   def forward(self, x):
#     return self.block(x) + self.shortcut(x)
