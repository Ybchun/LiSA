import numpy as np
import torch
from torch.utils import data


class sgloc_data(data.Dataset):
    def __init__(self, in_dataset, config):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.max_volume_space = config.max_volume_space
        self.min_volume_space = config.min_volume_space

    def __len__(self):
        'Denotes the total number of samples'

        return len(self.point_cloud_dataset)

    def get_scene(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        pose = data['pose']
        rot = data['rot']
        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        labels = labels[mask]
        # intensity = intensity[mask]
        point_num = len(xyz)

        data_dict = {}
        data_dict['point_feat'] = xyz
        data_dict['point_label'] = labels
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['pose'] = pose
        data_dict['rot'] = rot

        return data_dict

    def __getitem__(self, index):
        data_dict = self.get_scene(index)

        return data_dict


def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]
    pose = [torch.from_numpy(d['pose']) for d in data]

    return {
        'points': torch.cat(points).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).float(),
        'pose': torch.cat(pose).float(),
    }