from torch import nn


class RoIBoxPredictor(nn.Module):
    def __init__(self, num_classes):
        super(RoIBoxPredictor, self).__init__()
        representation_size = 512

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)


        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        ## fc layer
        representation_size_fc = 256
        self.cls_score_fc = nn.Linear(representation_size_fc, num_classes)
        self.bbox_pred_fc = nn.Linear(representation_size_fc, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score_fc.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_fc.weight, std=0.001)
        for l in [self.cls_score_fc, self.bbox_pred_fc]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):  # (x_reg, identity)
        ## conv cls
        # scores = self.cls_score(x[0])
        ## conv reg
        bbox_deltas = self.bbox_pred(x[0])

        x_fc_cls = x[1]
        # x_fc_reg = x[2]
        scores_fc = self.cls_score_fc(x_fc_cls)
        # bbox_deltas_fc = self.bbox_pred_fc(x_fc_reg)

        # return scores, bbox_deltas, scores_fc, bbox_deltas_fc
        return scores_fc, bbox_deltas
