import torch
import numpy as np
from torch_scatter import scatter_mean


def collate_fn_limit(batch, max_batch_points, logger):
    coord, xyz, feat, label = list(zip(*batch))
    offset, count = [], 0

    new_coord, new_xyz, new_feat, new_label = [], [], [], []
    k = 0
    for i, item in enumerate(xyz):

        count += item.shape[0]
        if count > max_batch_points:
            break

        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(label[i])

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in xyz])
        s_now = sum([x.shape[0] for x in new_xyz[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(
        new_label[:k]), torch.IntTensor(offset[:k])


def collation_fn_voxelmean(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, xyz, feats, inds_recons, file_path, poses = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)

    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    poses = torch.cat(poses)
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)

    return coords, xyz, feats, poses, offset, inds_recons, file_path


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        idx_start = np.cumsum(np.insert(count, 0, 0)[0:-1])
        idx_select = idx_start# + np.random.randint(0, count.max(), count.size) % count
        # idx_unique = idx_sort[idx_select]
        sorted_idx = np.zeros(key.shape[0]).astype(np.int)
        sorted_idx[idx_start] = 1
        sorted_idx = np.cumsum(sorted_idx) - 1
        idx_recon = np.zeros(key.shape[0]).astype(np.int)
        idx_recon[idx_sort] = sorted_idx
        return idx_recon


def data_prepare(coord, feat, label, rot, pose, split='train', voxel_size=np.array([0.1, 0.1, 0.1]), voxel_max=None):
    coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord_norm = coord - coord_min
    # 制作真值，需要存储世界坐标下的xyz
    gt = (rot @ feat.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
    if split == 'train':
        uniq_idx = voxelize(coord_norm, voxel_size)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        # 加入gt
        coord, feat, label, gt = coord[uniq_idx], feat[uniq_idx], label[uniq_idx], gt[uniq_idx]
        if voxel_max and label.shape[0] > voxel_max:
            init_idx = np.random.randint(label.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            coord, feat, label, gt = coord[crop_idx], feat[crop_idx], label[crop_idx], gt[crop_idx]
            coord_voxel = coord_voxel[crop_idx]
            # gtxyz mask
            label = np.concatenate((gt, label), axis=1)
        label = torch.LongTensor(label)
    else:
        idx_recon = voxelize(coord_norm, voxel_size, mode=1)

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    if split == 'train':
        coord_voxel = torch.LongTensor(coord_voxel)
        return coord_voxel, coord, feat, label
    else:
        # pose
        pose = torch.LongTensor(pose)
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coords_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        coord = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        return coords_voxel, coord, feat, idx_recon, pose