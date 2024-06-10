import torch
from torch import nn


# L1
class L1_CriterionCoordinate(nn.Module):
    def __init__(self):
        super(L1_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, pred_seg_feature, seg_feature):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)
        seg_loss_map = torch.sum(torch.abs(pred_seg_feature - seg_feature), axis=-1, keepdims=True)
        seg_loss_map = torch.mean(seg_loss_map)

        return 0.5 * loss_map + 0.5 * seg_loss_map


class DDPM_CriterionCoordinate(nn.Module):
    def __init__(self):
        super(DDPM_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, ddim_loss, kd_loss):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)

        return loss_map + ddim_loss + kd_loss


class CriterionCoordinate(nn.Module):
    def __init__(self):
        super(CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)

        return loss_map
