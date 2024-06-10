import torch
import torch_scatter
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F


# class SparseBasicBlock(spconv.SparseModule):
#     def __init__(self, in_channels, out_channels, kernel_size, indice_key):
#         super(SparseBasicBlock, self).__init__()
#         self.layers_in = spconv.SparseSequential(
#             spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
#             nn.BatchNorm1d(out_channels),
#         )
#         self.layers = spconv.SparseSequential(
#             spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
#             nn.BatchNorm1d(out_channels),
#             nn.LeakyReLU(inplace=True),
#             spconv.SubMConv3d(out_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
#             nn.BatchNorm1d(out_channels),
#         )
#
#     def forward(self, x):
#         identity = self.layers_in(x)
#         output = self.layers(x)
#         return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))


class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key):
        super(SparseBasicBlock, self).__init__()
        if in_channels == out_channels:
            self.layers_in = None
        else:
            self.layers_in =  spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            )

        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        residual = input
        output = self.layers(input)
        if self.layers_in is not None:
            residual = self.layers_in(input)
        output = output.replace_feature(self.relu(output.features + residual.features))
        return output


class ResidualBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.conv_branch = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, spatial_shape):
        super(ResConvBlock, self).__init__()
        self.indice_key = indice_key
        self.spatial_shape = spatial_shape
        # self.v_enc = ResidualBlock(in_channels, out_channels, kernel_size, self.indice_key)
        self.v_enc = SparseBasicBlock(in_channels, out_channels, kernel_size, self.indice_key)

    def forward(self, data_dict):
        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor'])

        # fusion and pooling
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=v_fea.features,
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )
        return data_dict


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, spatial_shape):
        super(ConvBlock, self).__init__()
        self.indice_key = indice_key
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
            )


    def forward(self, data_dict):
        # voxel encoder
        data_dict['sparse_tensor'] = self.v_enc(data_dict['sparse_tensor'])
        return data_dict


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, scale, last_scale, spatial_shape):
        super(DownBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        # self.v_enc = ResidualBlock(in_channels, out_channels, kernel_size, self.indice_key)
        self.v_enc = SparseBasicBlock(in_channels, out_channels, kernel_size, self.indice_key)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        # print("*****************************")
        # print(coors_inv_last.size())
        # print(data_dict['sparse_tensor'].features.size())
        # print(data_dict['sparse_tensor'].features[coors_inv_last].size())
        v_fea_inv = torch_scatter.scatter_mean(data_dict['sparse_tensor'].features[coors_inv_last], coors_inv, dim=0)

        # point encoder
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']
        # fusion and pooling
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=v_fea_inv,
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )
        # voxel encoder
        data_dict['sparse_tensor'] = self.v_enc(data_dict['sparse_tensor'])
        return data_dict


class RegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, spatial_shape):
        super(RegBlock, self).__init__()
        self.indice_key = indice_key
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=True)

    def forward(self, data_dict):
        return self.v_enc(data_dict['sparse_tensor'])


# TSFA
class Pool(nn.Module):
    # TSFA中降采样使用
    def __init__(self, scale, last_scale, spatial_shape):
        super(Pool, self).__init__()
        self.scale = scale
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape

    def forward(self, out, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        coors = data_dict['scale_{}'.format(self.scale)]['coors']
        v_fea = torch_scatter.scatter_mean(out.features[coors_inv_last], coors_inv, dim=0)
        out = spconv.SparseConvTensor(
            features=v_fea,
            indices=coors,
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )
        return out

class ResConv(nn.Module):
    # TSFA 残差卷积
    def __init__(self, in_channels, out_channels, kernel_size, indice_key):
        super(ResConv, self).__init__()
        self.indice_key = indice_key
        # self.v_enc = ResidualBlock(in_channels, out_channels, kernel_size, self.indice_key)
        self.v_enc = SparseBasicBlock(in_channels, out_channels, kernel_size, self.indice_key)

    def forward(self, out):
        return self.v_enc(out)

class Conv(nn.Module):
    # TSFA 卷积
    def __init__(self, in_channels, out_channels, kernel_size, indice_key):
        super(Conv, self).__init__()
        self.indice_key = indice_key
        self.v_enc = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, bias=False, indice_key=self.indice_key),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, out):
        return self.v_enc(out)