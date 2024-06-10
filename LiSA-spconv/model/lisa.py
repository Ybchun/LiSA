import torch
import torch_scatter
import torch.nn as nn
import numpy as np
from model.voxel_fea_generator import voxel_3d_generator, voxelization
from util.model_util import ResConvBlock, ConvBlock, DownBlock, RegBlock, Pool, Conv, ResConv
from model.sphereformer import Semantic
from model.loss import CriterionCoordinate
from kd_diffusion.diffkd import DiffKD
import torch.nn.functional as F


class attention_func(nn.Module):
    def __init__(self, l_ch, s_ch, m_ch, cat_ch, scale_list, strides, spatial_shape):
        super(attention_func, self).__init__()
        """
        l_ch: large feats 最上层特征维度
        s_ch: small feats 中间层特征维度
        m_ch: master feats 最下层特征维度
        """
        self.scale_list = scale_list
        self.strides = strides
        self.spatial_shape = spatial_shape
        self.l_pool = Pool(self.scale_list[2], self.scale_list[0], np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())
        self.s_pool = Pool(self.scale_list[2], self.scale_list[1], np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.l_conv = Conv(l_ch, cat_ch, kernel_size=1, indice_key='l_conv')
        self.s_conv = Conv(s_ch, cat_ch, kernel_size=1, indice_key='s_conv')

        self.l_squeeze = ResConv(cat_ch, 1, kernel_size=1, indice_key='l_squeeze')
        self.s_squeeze = ResConv(cat_ch, 1, kernel_size=1, indice_key='s_squeeze')
        self.m_squeeze = ResConv(m_ch, 1, kernel_size=1, indice_key='m_squeeze')

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, s_feats, l_feats, m_feats, data_dict):
        l_feats = self.l_pool(l_feats, data_dict)
        l_feats = self.l_conv(l_feats)
        s_feats = self.s_pool(s_feats, data_dict)
        s_feats = self.s_conv(s_feats)
        _s_feats = self.l_squeeze(s_feats)
        _l_feats = self.l_squeeze(l_feats)
        out_feats = self.m_squeeze(m_feats)
        att_map_sum = self.sigmoid(out_feats.features + _l_feats.features)
        att_map_mul = self.sigmoid(out_feats.features + _s_feats.features)
        out_feats = out_feats.replace_feature(torch.cat((m_feats.features, l_feats.features * att_map_sum,
                                                         s_feats.features * att_map_mul), 1))
        data_dict['sparse_tensor'] = out_feats
        return data_dict

class LiSA_ddpm(nn.Module):
    def __init__(self, config):
        super(LiSA_ddpm, self).__init__()
        "Initialization"
        self.input_dim = config.input_c
        self.output_dim = config.output_c
        self.att_dim = config.att_dim
        self.conv_dim = config.layers
        self.scale_list = config.scale_list
        self.num_scale = len(self.scale_list)
        min_volume_space = config.min_volume_space
        max_volume_space = config.max_volume_space
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config.spatial_shape)
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder
        self.conv1a = ConvBlock(in_channels=self.input_dim,
                                out_channels=self.conv_dim[0],
                                kernel_size=3,
                                indice_key='conv1a',
                                spatial_shape=np.int32(self.spatial_shape))

        self.conv1b = DownBlock(in_channels=self.conv_dim[0],
                                out_channels=self.conv_dim[1],
                                kernel_size=3,
                                indice_key='conv1b',
                                scale=self.scale_list[0],
                                last_scale=1,
                                spatial_shape=np.int32(self.spatial_shape // self.strides[0])[::-1].tolist())

        self.conv2a = DownBlock(in_channels=self.conv_dim[1],
                                out_channels=self.conv_dim[2],
                                kernel_size=3,
                                indice_key='conv2a',
                                scale=self.scale_list[1],
                                last_scale=self.scale_list[0],
                                spatial_shape=np.int32(self.spatial_shape // self.strides[1])[::-1].tolist())

        self.conv3a = ResConvBlock(in_channels=self.conv_dim[2],
                                   out_channels=self.conv_dim[3],
                                   kernel_size=3,
                                   indice_key='conv3a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[1])[::-1].tolist())

        self.conv3b = DownBlock(in_channels=self.conv_dim[3],
                                out_channels=self.conv_dim[4],
                                kernel_size=3,
                                indice_key='conv3b',
                                scale=self.scale_list[2],
                                last_scale=self.scale_list[1],
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4a = ResConvBlock(in_channels=self.conv_dim[4],
                                   out_channels=self.conv_dim[5],
                                   kernel_size=3,
                                   indice_key='conv4a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4b = ResConvBlock(in_channels=self.conv_dim[5],
                                   out_channels=self.conv_dim[6],
                                   kernel_size=3,
                                   indice_key='conv4b',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4c = ResConvBlock(in_channels=self.conv_dim[6],
                                   out_channels=self.conv_dim[7],
                                   kernel_size=3,
                                   indice_key='conv4c',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv5a = ResConvBlock(in_channels=self.conv_dim[7] + 2 * self.att_dim,
                                   out_channels=self.conv_dim[8],
                                   kernel_size=1,
                                   indice_key='conv5a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv5b = ResConvBlock(in_channels=self.conv_dim[8],
                                   out_channels=self.conv_dim[9],
                                   kernel_size=1,
                                   indice_key='conv5b',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.convout = RegBlock(in_channels=self.conv_dim[9],
                                out_channels=self.output_dim,
                                indice_key='convout',
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())
        self.convout_seg = RegBlock(in_channels=self.conv_dim[9],
                                out_channels=32,
                                indice_key='convout_seg',
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.msf = attention_func(128, 256, 512, 128, self.scale_list, self.strides, self.spatial_shape)

        self.diffkd = DiffKD(student_channels=32, teacher_channels=32,
                    kernel_size=1, use_ae=False)

    def reshape_(self, x):
        return x.view(x.shape[0], x.shape[1], 1, 1)

    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)
        # encoder
        data_dict = self.conv1a(data_dict)
        # print("conv1a", list(data_dict.keys()))
        out1b = self.conv1b(data_dict)['sparse_tensor']
        # print("conv1b", list(data_dict.keys()))
        # print("out1b", out1b.indices.size(), out1b.features.size())
        data_dict = self.conv2a(data_dict)
        # print("conv2a", list(data_dict.keys()))
        # exit(1)
        out3a = self.conv3a(data_dict)['sparse_tensor']
        data_dict = self.conv3b(data_dict)
        data_dict = self.conv4a(data_dict)
        data_dict = self.conv4b(data_dict)
        out = self.conv4c(data_dict)['sparse_tensor']
        # # multi-scale
        data_dict = self.conv5a(self.msf(out3a, out1b, out, data_dict))
        data_dict = self.conv5b(data_dict)
        out = self.convout(data_dict)
        seg_out = self.convout_seg(data_dict)

        data_dict['out'] = out
        # data_dict['labels'] = self.voxelize_labels(data_dict)

        teacher_seg_feature = data_dict['labels'][:, 6:]
        student_seg_feature = seg_out.features
        student_feat_refined, ddim_loss, teacher_feat, rec_loss = \
            self.diffkd(self.reshape_(student_seg_feature), self.reshape_(teacher_seg_feature))
        kd_loss = F.mse_loss(student_feat_refined, teacher_feat)
        gt_xyz = data_dict['labels'][:, :6]
        return out.features, gt_xyz, ddim_loss, kd_loss, data_dict['coors']

class LiSA(nn.Module):
    def __init__(self, config):
        super(LiSA, self).__init__()
        "Initialization"
        self.input_dim = config.input_c
        self.output_dim = config.output_c
        self.att_dim = config.att_dim
        self.conv_dim = config.layers
        self.scale_list = config.scale_list
        self.num_scale = len(self.scale_list)
        min_volume_space = config.min_volume_space
        max_volume_space = config.max_volume_space
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config.spatial_shape)
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder
        self.conv1a = ConvBlock(in_channels=self.input_dim,
                                out_channels=self.conv_dim[0],
                                kernel_size=3,
                                indice_key='conv1a',
                                spatial_shape=np.int32(self.spatial_shape))

        self.conv1b = DownBlock(in_channels=self.conv_dim[0],
                                out_channels=self.conv_dim[1],
                                kernel_size=3,
                                indice_key='conv1b',
                                scale=self.scale_list[0],
                                last_scale=1,
                                spatial_shape=np.int32(self.spatial_shape // self.strides[0])[::-1].tolist())

        self.conv2a = DownBlock(in_channels=self.conv_dim[1],
                                out_channels=self.conv_dim[2],
                                kernel_size=3,
                                indice_key='conv2a',
                                scale=self.scale_list[1],
                                last_scale=self.scale_list[0],
                                spatial_shape=np.int32(self.spatial_shape // self.strides[1])[::-1].tolist())

        self.conv3a = ResConvBlock(in_channels=self.conv_dim[2],
                                   out_channels=self.conv_dim[3],
                                   kernel_size=3,
                                   indice_key='conv3a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[1])[::-1].tolist())

        self.conv3b = DownBlock(in_channels=self.conv_dim[3],
                                out_channels=self.conv_dim[4],
                                kernel_size=3,
                                indice_key='conv3b',
                                scale=self.scale_list[2],
                                last_scale=self.scale_list[1],
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4a = ResConvBlock(in_channels=self.conv_dim[4],
                                   out_channels=self.conv_dim[5],
                                   kernel_size=3,
                                   indice_key='conv4a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4b = ResConvBlock(in_channels=self.conv_dim[5],
                                   out_channels=self.conv_dim[6],
                                   kernel_size=3,
                                   indice_key='conv4b',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4c = ResConvBlock(in_channels=self.conv_dim[6],
                                   out_channels=self.conv_dim[7],
                                   kernel_size=3,
                                   indice_key='conv4c',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv5a = ResConvBlock(in_channels=self.conv_dim[7] + 2 * self.att_dim,
                                   out_channels=self.conv_dim[8],
                                   kernel_size=1,
                                   indice_key='conv5a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv5b = ResConvBlock(in_channels=self.conv_dim[8],
                                   out_channels=self.conv_dim[9],
                                   kernel_size=1,
                                   indice_key='conv5b',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.convout = RegBlock(in_channels=self.conv_dim[9],
                                out_channels=self.output_dim,
                                indice_key='convout',
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())
        self.convout_seg = RegBlock(in_channels=self.conv_dim[9],
                                out_channels=32,
                                indice_key='convout_seg',
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.msf = attention_func(128, 256, 512, 128, self.scale_list, self.strides, self.spatial_shape)

    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)
        # print("voxel_3d_generator", list(data_dict.keys()))

        # encoder
        data_dict = self.conv1a(data_dict)
        # print("conv1a", list(data_dict.keys()))
        out1b = self.conv1b(data_dict)['sparse_tensor']
        # print("conv1b", list(data_dict.keys()))
        # print("out1b", out1b.indices.size(), out1b.features.size())
        data_dict = self.conv2a(data_dict)
        # print("conv2a", list(data_dict.keys()))
        # exit(1)
        out3a = self.conv3a(data_dict)['sparse_tensor']
        data_dict = self.conv3b(data_dict)
        data_dict = self.conv4a(data_dict)
        data_dict = self.conv4b(data_dict)
        out = self.conv4c(data_dict)['sparse_tensor']
        # # multi-scale
        data_dict = self.conv5a(self.msf(out3a, out1b, out, data_dict))
        data_dict = self.conv5b(data_dict)
        out = self.convout(data_dict)
        seg_out = self.convout_seg(data_dict)

        return out.features, seg_out.features, data_dict['labels'], data_dict['coors']
