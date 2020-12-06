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

# class DeformConv2d(nn.Module):
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
#         """
#         Args:
#             modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
#         """
#         super(DeformConv2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.zero_padding = nn.ZeroPad2d(padding)
#         self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

#         self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#         nn.init.constant_(self.p_conv.weight, 0)
#         self.p_conv.register_backward_hook(self._set_lr)

#         self.modulation = modulation
#         if modulation:
#             self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#             nn.init.constant_(self.m_conv.weight, 0)
#             self.m_conv.register_backward_hook(self._set_lr)

#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
#         grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

#     def forward(self, x):
#         offset = self.p_conv(x)
#         if self.modulation:
#             m = torch.sigmoid(self.m_conv(x))

#         dtype = offset.data.type()
#         ks = self.kernel_size
#         N = offset.size(1) // 2

#         if self.padding:
#             x = self.zero_padding(x)

#         # (b, 2N, h, w)
#         p = self._get_p(offset, dtype)

#         # (b, h, w, 2N)
#         p = p.contiguous().permute(0, 2, 3, 1)
#         q_lt = p.detach().floor()
#         q_rb = q_lt + 1

#         q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

#         # clip p
#         p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

#         # bilinear kernel (b, h, w, N)
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

#         # (b, c, h, w, N)
#         x_q_lt = self._get_x_q(x, q_lt, N)
#         x_q_rb = self._get_x_q(x, q_rb, N)
#         x_q_lb = self._get_x_q(x, q_lb, N)
#         x_q_rt = self._get_x_q(x, q_rt, N)

#         # (b, c, h, w, N)
#         x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
#                    g_rb.unsqueeze(dim=1) * x_q_rb + \
#                    g_lb.unsqueeze(dim=1) * x_q_lb + \
#                    g_rt.unsqueeze(dim=1) * x_q_rt

#         # modulation
#         if self.modulation:
#             m = m.contiguous().permute(0, 2, 3, 1)
#             m = m.unsqueeze(dim=1)
#             m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
#             x_offset *= m

#         x_offset = self._reshape_x_offset(x_offset, ks)
#         out = self.conv(x_offset)

#         return out

#     def _get_p_n(self, N, dtype):
#         p_n_x, p_n_y = torch.meshgrid(
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
#         # (2N, 1)
#         p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
#         p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

#         return p_n

#     def _get_p_0(self, h, w, N, dtype):
#         p_0_x, p_0_y = torch.meshgrid(
#             torch.arange(1, h*self.stride+1, self.stride),
#             torch.arange(1, w*self.stride+1, self.stride))
#         p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

#         return p_0

#     def _get_p(self, offset, dtype):
#         N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

#         # (1, 2N, 1, 1)
#         p_n = self._get_p_n(N, dtype)
#         # (1, 2N, h, w)
#         p_0 = self._get_p_0(h, w, N, dtype)
#         p = p_0 + p_n + offset
#         return p

#     def _get_x_q(self, x, q, N):
#         b, h, w, _ = q.size()
#         padded_w = x.size(3)
#         c = x.size(1)
#         # (b, c, h*w)
#         x = x.contiguous().view(b, c, -1)

#         # (b, h, w, N)
#         index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
#         # (b, c, h*w*N)
#         index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

#         x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

#         return x_offset

#     @staticmethod
#     def _reshape_x_offset(x_offset, ks):
#         b, c, h, w, N = x_offset.size()
#         x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
#         x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

#         return x_offset

class AdaptiveConv(nn.Module):
    """ Adaptive Conv is built based on Deformable Conv
    with precomputed offsets which derived from anchors"""

    def __init__(self, in_channels, out_channels, dilation=1, adapt=False):
        super(AdaptiveConv, self).__init__()
        self.adapt = adapt
        if self.adapt:
            assert dilation == 1
            self.conv = DeformConv2d(in_channels, out_channels, 3, padding=1)
        else:  # fallback to normal Conv2d
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation)
        # print(self.conv.weight)
    # def init_weights(self):
    #     nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x, offset):
        if self.adapt:
            import pdb
            pdb.set_trace()
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
        # self.rpn_conv = DeformConv2d(in_channels, feat_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(feat_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            feat_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in self.children():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)

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


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # type: (List[Tensor])
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg