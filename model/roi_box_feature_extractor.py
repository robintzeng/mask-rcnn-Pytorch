import torch.nn as nn
from poolers import Pooler
import resnet
# from torchvision.models import resnet


class ROIFeatureExtractor(nn.Module):  # ResNet50 Conv5
    def __init__(self):
        super(ROIFeatureExtractor, self).__init__()

        pooler = Pooler(
            output_size=(14, 14),  # resolution = 14
            scales=(1.0 / 16,),
            sampling_ratio=0,  # TODO !
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module="BottleneckWithFixedBatchNorm",
            stages=(stage,),
            num_groups=1,
            width_per_group=64,
            stride_in_1x1=True,
            stride_init=None,
            res2_out_channels=256,
            dilation=1
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x