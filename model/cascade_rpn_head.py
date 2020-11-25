import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class DeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups)

class AdaptiveConv(nn.Module):
    """ Adaptive Conv is built based on Deformable Conv
    with precomputed offsets which derived from anchors"""

    def __init__(self, in_channels, out_channels, dilation=1, adapt=False):
        super(AdaptiveConv, self).__init__()
        self.adapt = adapt
        if self.adapt:
            assert dilation == 1
            self.conv = DeformConv(in_channels, out_channels, 3, padding=1)
        else:  # fallback to normal Conv2d
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation)
        # print(self.conv.weight)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x, offset):
        if self.adapt:
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            x = self.conv(x, offset)
        else:
            # assert offset is None
            # print(offset)
            x = self.conv(x)
        return x

class CascadeRPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    # feat_channels = 256
    def __init__(self, in_channels, feat_channels, num_anchors, stage=3):
        super(CascadeRPNHead, self).__init__()
        self.stage = stage
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.rpn_conv = AdaptiveConv(in_channels, feat_channels, dilation=1, adapt=False)
        self.cls_logits = nn.Conv2d(feat_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            feat_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        # for l in self.children():
        #     torch.nn.init.normal_(l.weight, std=0.01)
        #     torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # type: (List[Tensor])
        logits = []
        bbox_reg = []
        for feature in x:
            feature_step = feature * 1
            for i in range(self.stage):
                feature_step = F.relu(self.conv(feature_step))
                cls_logits = self.cls_logits(feature_step)
                bbox_pred = self.bbox_pred(feature_step)
                feature_step = F.relu(self.rpn_conv(feature_step, bbox_pred))   
            logits.append(cls_logits)
            bbox_reg.append(bbox_pred)
        return logits, bbox_reg

    # def forward(self, x):
    #     # type: (List[Tensor])
    #     logits = []
    #     bbox_reg = []
    #     for feature in x:
    #         for i in range(self.stage):
    #             t = F.relu(self.rpn_conv(feature))
    #         logits.append(self.cls_logits(t))
    #         bbox_reg.append(self.bbox_pred(t))
    #     return logits, bbox_reg