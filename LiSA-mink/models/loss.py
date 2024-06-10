import torch
from torch import nn
import torch.nn.functional as F
from kd_diffusion.diffkd import DiffKD

# L1
class CriterionCoordinate(nn.Module):
    def __init__(self):
        super(CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)

        return loss_map


class Seg_CriterionCoordinate(nn.Module):
    def __init__(self):
        super(Seg_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, pred_seg_feature, seg_feature):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)
        seg_loss_map = torch.sum(torch.abs(pred_seg_feature - seg_feature), axis=-1, keepdims=True)
        seg_loss_map = torch.mean(seg_loss_map)

        return 0.5 * loss_map + 0.5 * seg_loss_map

# class Seg_DDPM_CriterionCoordinate(nn.Module):
#     def __init__(self) -> None:
#         super(Seg_DDPM_CriterionCoordinate, self).__init__()
#         self.diffkd = DiffKD(
#             student_channels=32, teacher_channels=32, kernel_size=1, use_ae=False
#         )

#     def _reshape(self, x):
#         return x.view(x.shape[0], x.shape[1], 1, 1)

#     def forward(self, pred_point, gt_point, pred_seg_feature, seg_feature):
#         # self.diffkd = self.diffkd.to(pred_point.device)
#         # print(pred_point.device)
#         loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
#         loss_map = torch.mean(loss_map)

#         # print("==================================")
#         # print("pred_seg_feature:", type(pred_seg_feature))
#         # print("seg_feature:", type(seg_feature))
#         # print("pred_seg_feature:", pred_seg_feature.size())
#         # print("seg_feature:", seg_feature.size())
#         # exit(1)

#         student_feat_refined, ddim_loss, teacher_feat, _ = self.diffkd(
#             self._reshape(pred_seg_feature), self._reshape(seg_feature)
#         )
#         kd_loss = F.mse_loss(student_feat_refined, teacher_feat)

#         return loss_map + ddim_loss + kd_loss

class Seg_DDPM_CriterionCoordinate(nn.Module):
    def __init__(self) -> None:
        super(Seg_DDPM_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, ddim_loss, kd_loss):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)


        return loss_map + ddim_loss + kd_loss


class IGCC_Seg_CriterionCoordinate(nn.Module):
    def __init__(self):
        super(IGCC_Seg_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, pred_seg_feature, seg_feature):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        # print("11111111111", loss_map.size())
        loss_map = torch.mean(loss_map)
        # print("22222222222", loss_map)
        pred_pairwise_distance = torch.sum(torch.abs(pred_point[:, None, :] - pred_point[None, :, :]), dim=-1)
        gt_pairwise_distance = torch.sum(torch.abs(gt_point[:, None, :] - gt_point[None, :, :]), dim=-1)
        # print("3333333333333", pred_pairwise_distance.size())
        distance_pair_mask = torch.abs(gt_pairwise_distance - pred_pairwise_distance) < 3  # 3
        # print("4444444444444", distance_pair_mask.numel())
        igcc_loss = torch.sum(torch.abs(pred_pairwise_distance - gt_pairwise_distance)) / distance_pair_mask.numel()
        # print("5555555555555", igcc_loss)
        seg_loss_map = torch.sum(torch.abs(pred_seg_feature - seg_feature), axis=-1, keepdims=True)
        # print("6666666666666", seg_loss_map.size())
        seg_loss_map = torch.mean(seg_loss_map)
        # print("7777777777777", seg_loss_map)

        return loss_map + igcc_loss + seg_loss_map

# Plane L1
class Plane_CriterionCoordinate(nn.Module):
    def __init__(self):
        super(Plane_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, mask):
        diff_coord_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = mask * diff_coord_map
        valid_coord = torch.sum(mask)
        loss_map = (torch.sum(loss_map) + 1e-9) / (valid_coord + 1e-9)
        return loss_map


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred_point, gt_point, mask):
        pred_pairwise_distance = torch.sum(torch.abs(pred_point[:, None, :] - pred_point[None, :, :]), dim=-1)
        gt_pairwise_distance = torch.sum(torch.abs(gt_point[:, None, :] - gt_point[None, :, :]), dim=-1)
        plane_pair_mask = mask @ mask.T
        distance_pair_mask = torch.abs(gt_pairwise_distance - pred_pairwise_distance) < 3       # 3
        pair_mask = plane_pair_mask*distance_pair_mask
        loss = torch.sum(pair_mask*torch.abs(pred_pairwise_distance - gt_pairwise_distance))
        # num用于计算有多少个可用元素
        valid_coord  = torch.sum(pair_mask)

        return loss, valid_coord


# IGCC
class Neighborhood_Distance(nn.Module):
    def __init__(self):
        super(Neighborhood_Distance, self).__init__()
        self.edge_loss = EdgeLoss()
        self.node_loss = Plane_CriterionCoordinate()

    def forward(self, pred_point, gt_point, mask, index):
        # node_loss = 0.0
        edge_loss = 0.0
        valid_coord       = 0
        # n = len(pred_point)
        for i in range(len(index)-1):
            batch_pred_point = pred_point[index[i]:index[i+1], :].view(-1, 3)
            batch_gt_point   = gt_point[index[i]:index[i+1], :].view(-1, 3)
            batch_mask       = mask[index[i]:index[i+1],:].view(-1, 1)
            # node_loss       += self.node_loss(batch_pred_point, batch_gt_point, batch_mask)
            batch_edge_loss, batch_valid_coord = self.edge_loss(batch_pred_point, batch_gt_point, batch_mask)
            edge_loss        += batch_edge_loss
            valid_coord      += batch_valid_coord

        node_loss = self.node_loss(pred_point, gt_point, mask)
        if valid_coord >= 1:
            edge_loss = edge_loss / valid_coord
            return node_loss + edge_loss
        else:
            return node_loss







