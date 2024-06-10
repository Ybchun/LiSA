#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: voxel_fea_generator.py
@time: 2021/8/4 13:36
'''
import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        # print("pc:", pc[:10, :])
        # print(self.scale_list)
        for idx, scale in enumerate(self.scale_list):
            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale))
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))

            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx, yidx, zidx], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }

            # print(scale)
            # print(data_dict['scale_{}'.format(scale)]['full_coors'].shape)  # 生成的坐标，含重复
            # print(data_dict['scale_{}'.format(scale)]['coors_inv'].shape)  # 每个坐标重复的次数
            # print(data_dict['scale_{}'.format(scale)]['coors'].shape)  # 生成的坐标，去重
            # print(data_dict['scale_{}'.format(scale)]['coors'])
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz

    def forward(self, data_dict):
        features = torch_scatter.scatter_mean(data_dict['points'], data_dict['scale_1']['coors_inv'], dim=0)
        labels = torch_scatter.scatter_mean(data_dict['labels'], data_dict['scale_8']['coors_inv'], dim=0)
        # intensity = torch_scatter.scatter_mean(data_dict['intensity'], data_dict['scale_1']['coors_inv'], dim=0)

        data_dict['labels'] = labels
        # data_dict['feature32'] = labels[:, 6:]
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']
        # sphereformer
        # data_dict['seg_spare_tensor'] = spconv.SparseConvTensor(
        #     features=torch.cat((data_dict['scale_1']['coors'][:, 1:], intensity), dim=-1),
        #     indices=data_dict['scale_1']['coors'].int(),
        #     spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
        #     batch_size=data_dict['batch_size']
        # )
        # data_dict['seg_xyz'] = data_dict['scale_1']['coors'][:, 1:]
        # data_dict['seg_batch'] = data_dict['scale_1']['coors'][:, 0]
        # del data_dict['intensity']

        return data_dict