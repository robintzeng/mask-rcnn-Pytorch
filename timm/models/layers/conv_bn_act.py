""" Conv2d + BN + Act

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn.functional as F
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import convert_norm_act_type
import torch.nn.init as init
from torch.nn.modules.utils import _pair
import math
class AttentionConv(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
    #     super(AttentionConv, self).__init__()
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels
    #     self.kernel_size = kernel_size
    #     self.stride = stride
    #     self.padding = padding
    #     self.groups = groups

    #     assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

    #     self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
    #     self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

    #     self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    #     self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    #     self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    #     self.reset_parameters()

    # def forward(self, x):
    #     batch, channels, height, width = x.size()
    #     # print("ZZZZZZZZZZZZZZZZZZZZZZ")
    #     # print(x.size())
    #     # print("ZZZZZZZZZZZZZZZZ")
    #     padded_x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
    #     # print("PPPPPPPPPPPPPP")
    #     # print(padded_x.size())
    #     # print("PPPPPPPPPPPPP")
    #     q_out = self.query_conv(x)
    #     k_out = self.key_conv(padded_x)
    #     v_out = self.value_conv(padded_x)
    #     # print(q_out.size())
    #     # print(k_out.size())
    #     # print(v_out.size())

    #     k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
    #     v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
    #     # print(k_out.size())
    #     # print(v_out.size())

    #     k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
    #     k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        

    #     k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
    #     v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

    #     q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

    #     out = q_out * k_out
    #     out = F.softmax(out, dim=-1)
    #     out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

    #     return out
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias=True):
        super(AttentionConv, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups # multi-head count

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0]))

        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_value = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)

        self.softmax = nn.Softmax(dim=3)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_key.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_value.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        init.normal_(self.relative_x, 0, 1)
        init.normal_(self.relative_y, 0, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2

        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        vq = self.weight_query(x)
        vk = self.weight_key(x)
        vv = self.weight_value(x) # b, fc, ph, pw

        # b, fc, fh, fw
        win_q = vq[:, :, (kh-1)//2:ph-(kh//2):self.stride[0], (kw-1)//2:pw-(kw//2):self.stride[1]]

        win_q_b = win_q.view(b, self.groups, -1, fh, fw) # b, g, fc/g, fh, fw

        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=2) # (b, g, x, fh, fw), (b, g, y, fh, fw)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.relative_x)) # b, fh, fw, kw
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.relative_y)) # b, fh, fw, kh

        win_k = vk.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1]) # b, fc, fh, fw, kh, kw

        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(dim=1)  # b, fh, fw, kh, kw
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4) # add rel_x, rel_y
        vx = self.softmax(vx.view(b, fh, fw, -1)).view(b, 1, fh, fw, kh, kw)

        win_v = vv.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v, )) # (b, fc, fh, fw, kh, kw) -> (b, fc, fh, fw)

        if self.bias is not None:
            fin_v += self.bias

        return fin_v

    # def reset_parameters(self):
    #     init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
    #     init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
    #     init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

    #     init.normal_(self.rel_h, 0, 1)
    #     init.normal_(self.rel_w, 0, 1)

class AttnConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=nn.ReLU, apply_act=True,
                 drop_block=None, aa_layer=None):
        super(AttnConvBnAct, self).__init__()
        use_aa = aa_layer is not None

        self.conv = AttentionConv(in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
             padding = padding ,groups=groups, bias=False)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=nn.ReLU, apply_act=True,
                 drop_block=None, aa_layer=None):
        super(ConvBnAct, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x
