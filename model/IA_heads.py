import torch
import torch.nn.functional as F

from torchvision.models.detection.roi_heads import * #RoIHeads, fastrcnn_loss
from torch.autograd import Variable
from torch.jit.annotations import Optional, List, Dict, Tuple

class IA_roi_heads(RoIHeads):
  def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

       
        ### feature tensor F: (num_rois_per_img*num_images) * num_channels * 7 * 7 ###
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        ################################################################
        if self.training:
            self.eval()
            box_features_tmp = box_features.clone().detach()
            box_features_tmp = Variable(box_features_tmp.data, requires_grad=True)
            box_features_new = self.box_head(box_features_tmp)
            output, _ = self.box_predictor(box_features_new)
            class_num = output.shape[1]
            index = torch.cat(tuple(labels),0)#labels
            num_rois = box_features_tmp.shape[0]
            num_channel = box_features_tmp.shape[1]
            one_hot = torch.zeros((1), dtype=torch.float32).to(box_features.device)#.cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().to(box_features.device)#.cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)  # [n, 21]
            ### Get the classification score only on the ground-truth category ###
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            ### gradient tensor G: (num_rois_per_img*num_images) * num_channels * 7 * 7 ###
            grads_val = box_features_tmp.grad.clone().detach()
            ### global pooling to produce weight vector w: (num_rois_per_img*num_images) * num_channels * 1 ###
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            ### gradient guided attetion map M: (num_rois_per_img*num_images) * 1 * 7 * 7 ###
            cam_all = torch.sum(box_features_tmp * grad_channel_mean.view(num_rois, num_channel, 1, 1), 1)
            cam_all = cam_all.view(num_rois, 49)
            # cam_all = cam_all.view(num_rois, -1)
            self.zero_grad()

            # -------------------------IA ----------------------------
            num_s = 18
            ### Get a threshold T_s
            th_mask_value = torch.sort(cam_all, dim=1, descending=True)[0][:, num_s] # (num_rois_per_img*num_images)
            th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, 49) # (num_rois_per_img*num_images) * 49
            # th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, cam_all.shape[1])
            ### The spatial-wise inverted attention map A^s ###
            mask_all_cuda = torch.where(cam_all > th_mask_value, torch.zeros(cam_all.shape).to(box_features.device), torch.ones(cam_all.shape).to(box_features.device))
            mask_all = mask_all_cuda.reshape(num_rois, 7, 7).view(num_rois, 1, 7, 7)

            # ------------------------ batch ---------------------
            box_features_before_after = torch.cat((box_features_tmp, box_features_tmp * mask_all), dim=0) # (num_rois_per_img*num_images*2) * num_channels * 7 * 7
            box_features_before_after = self.box_head(box_features_before_after)

            cls_score_before_after, _ = self.box_predictor(box_features_before_after) # (num_rois_per_img*num_images*2) * num_classes
            cls_prob_before_after = F.softmax(cls_score_before_after, dim=1) # (num_rois_per_img*num_images*2) * num_classes
             
            cls_prob_before = cls_prob_before_after[0: num_rois] # (num_rois_per_img*num_images) * num_classes
            cls_prob_after = cls_prob_before_after[num_rois: num_rois * 2] # (num_rois_per_img*num_images) * num_classes, scores from where has small attention

            prepare_mask_fg_num = index.nonzero().size(0)#labels.nonzero().size(0)
            prepare_mask_bg_num = num_rois - prepare_mask_fg_num

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().to(box_features.device)#.cuda()
            ### Get the classification score only on the ground-truth category ###
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1) # (num_rois_per_img*num_images)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1) # (num_rois_per_img*num_images)
            change_vector = before_vector - after_vector - 0.01 
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).to(box_features.device)) # (num_rois_per_img*num_images)

            # fg_index = torch.where(labels > 0, torch.ones(change_vector.shape).to(box_features.device), torch.zeros(change_vector.shape).to(box_features.device))
            ### mask, not background = 1, else = 0 ###
            fg_index = torch.where(index > 0, torch.ones(change_vector.shape).to(box_features.device), torch.zeros(change_vector.shape).to(box_features.device))
            bg_index = 1 - fg_index
            if fg_index.nonzero().shape[0] != 0:
                not_01_fg_index = fg_index.nonzero()[:, 0].long() # indices of rois that aren't background
            else:
                not_01_fg_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to(box_features.device).long()
            not_01_bg_index = bg_index.nonzero()[:, 0].long() # indices of rois that are background
            change_vector_fg = change_vector[not_01_fg_index]
            change_vector_bg = change_vector[not_01_bg_index]

            for_fg_change_vector = change_vector.clone()
            for_bg_change_vector = change_vector.clone()
            for_fg_change_vector[not_01_bg_index] = -10000
            for_bg_change_vector[not_01_fg_index] = -10000
          
            th_fg_value = torch.sort(change_vector_fg, dim=0, descending=True)[0][int(round(float(prepare_mask_fg_num) / 5))]
            drop_index_fg = for_fg_change_vector.gt(th_fg_value)
            th_bg_value = torch.sort(change_vector_bg, dim=0, descending=True)[0][int(round(float(prepare_mask_bg_num) / 30))]
            drop_index_bg = for_bg_change_vector.gt(th_bg_value)
            drop_index_fg_bg = drop_index_fg + drop_index_bg
            ignore_index_fg_bg = drop_index_fg_bg.logical_not()
            # ignore_index_fg_bg = 1 - drop_index_fg_bg
            not_01_ignore_index_fg_bg = ignore_index_fg_bg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg_bg.long(), :] = 1

            # ---------------------------------------------------------
            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            box_features = box_features * mask_all
        ################################################################

        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses
