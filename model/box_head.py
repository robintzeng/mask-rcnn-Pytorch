import torch.nn as nn
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.resnet import BasicBlock, Bottleneck

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class RoIBoxHead(nn.Module):
    def __init__(self, in_features, num_classes, loss_weights=[1, 1, 1, 1], pretrained=False):
        super(RoIBoxHead, self).__init__()
        self.fc_head = TwoMLPHead(in_channels=1280*7*7, representation_size=in_features)
        layers = [
            ResidualBlock(256*5, 1024*5), ResidualBottleneckBlock(1024*5, 1024*5),
            ResidualBlock(256*5, 1024*5), ResidualBottleneckBlock(1024*5, 1024*5),
            ResidualBlock(256*5, 1024*5), ResidualBottleneckBlock(1024*5, 1024*5)
            ]
        self.conv_head = nn.Sequential(*layers)
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        if pretrained:
          self.fc_head.load_state_dict('state_dict_path')
          self.conv_head.load_state_dict('state_dict_path')


    def forward(self, features):  # N, 1280, 7, 7
        # class_logits, box_regression, class_logits_fc, box_regression_fc = self.predictor(x)
        fc_feature = self.fc_head.forward(features)
        conv_feature = self.conv_head(features)
        avgPool = nn.AvgPool2d((conv_feature.shape[2], conv_feature.shape[3]))
        conv_feature = avgPool(conv_feature)

        return (fc_feature, conv_feature)



class RoIBoxPredictor(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(RoIBoxPredictor, self).__init__()
        
        self.cls_score = nn.Linear(1024, num_classes)
        num_bbox_reg_classes = num_classes

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        self.bbox_pred = nn.Linear(1024, num_bbox_reg_classes * 4)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        if pretrained:
          self.cls_score.load_state_dict('state_dict_path')
          self.bbox_pred.load_state_dict('state_dict_path')
        

    def forward(self, two_features):
        fc_feature, conv_feature = two_features
        cls_logit = self.cls_score(fc_feature)
        bbox_pred = self.bbox_pred(conv_feature)

        return cls_logit, bbox_pred



'''
**********************************************************************************
'''



class PlainBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()
    self.net = nn.Sequential(
      nn.BatchNorm2d(Cin),
      nn.ReLU(),
      nn.Conv2d(Cin, Cout, 3, stride=1+int(downsample), padding=1),
      nn.BatchNorm2d(Cout),
      nn.ReLU(),
      nn.Conv2d(Cout, Cout, 1, padding=1)
    )

  def forward(self, x):
    return self.net(x)


class ResidualBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()
    self.block = PlainBlock(Cin, Cout, downsample)
    if downsample:
      self.shortcut = nn.Conv2d(Cin, Cout, 1, stride=2)
    else:
      self.shortcut = nn.Identity() if Cin == Cout else nn.Conv2d(Cin, Cout, 1)

  def forward(self, x):
    return self.block(x) + self.shortcut(x)


class ResidualBottleneckBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()
    self.block = nn.Sequential(
      nn.BatchNorm2d(Cin),
      nn.ReLU(),
      nn.Conv2d(Cin, Cout // 4, 1),
      nn.BatchNorm2d(Cout // 4),
      nn.ReLU(),
      nn.Conv2d(Cout // 4, Cout // 4, 3, padding=1),
      nn.BatchNorm2d(Cout // 4),
      nn.ReLU(),
      nn.Conv2d(Cout // 4, Cout, 1)
    )
    self.shortcut = nn.Identity() if Cin == Cout else nn.Conv2d(Cin, Cout, 1)

  def forward(self, x):
    return self.block(x) + self.shortcut(x)
