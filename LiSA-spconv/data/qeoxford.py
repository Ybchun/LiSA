import os
import numpy as np
import os.path as osp
import h5py
import random
import math
from torch.utils import data
import transforms3d.quaternions as txq
from util.pose_util import calibrate_process_poses, filter_overflow_ts, qlog
from data.robotcar_sdk.python.velodyne import load_velodyne_binary_seg, load_velodyne_binary, load_velodyne_binary_seg_feature32

BASE_DIR = osp.dirname(osp.abspath(__file__))


class QEOxford(data.Dataset):
    def __init__(self, data_path, split='train', real=False, num_grid_x=1, num_grid_y=1, block_num=1, augment=False):
        # directories
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'Oxford')
        extrinsics_dir = osp.join(BASE_DIR, 'robotcar_sdk', 'extrinsics')

        # decide which sequences to use
        if split == 'train':
            split_filename = osp.join(data_dir, 'train_split.txt')
            self.train = True
        else:
            split_filename = osp.join(data_dir, 'valid_split.txt')
            self.train = False
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        pcs_all = []
        self.pcs = []
        # 位姿中已加入外参
        print("I'M QEOxford")
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_calibrate' + str(real) + '.h5')
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps')
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
                rot = np.fromfile(osp.join(seq_dir, 'rot_tr.bin'), dtype=np.float32).reshape(-1, 9)
                t = np.fromfile(osp.join(seq_dir, 'tr_add_mean.bin'), dtype=np.float32).reshape(-1, 3)
                ps[seq] = np.concatenate((rot, t), axis=1)  # (n, 12)
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            if self.train:
                pcs_all.extend(
                    [osp.join(seq_dir, 'sphere_velodyne_left_feature32', '{:d}.bin'.format(t)) for t in ts[seq]])
            else:
                pcs_all.extend(
                    [osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'QEOxford_pose_stats.txt')
        if split=='train':
            mean_t = np.mean(poses[:, 9:], axis=0)  # (3,)
            std_t = np.std(poses[:, 9:], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))
        poses_all = np.empty((0, 6))
        rots_all = np.empty((0, 3, 3))

        pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')

        for seq in seqs:
            pss, rotation, pss_max, pss_min = calibrate_process_poses(poses_in=ps[seq], mean_t=mean_t,
                                                                      align_R=vo_stats[seq]['R'],
                                                                      align_t=vo_stats[seq]['t'],
                                                                      align_s=vo_stats[seq]['s'])
            poses_all = np.vstack((poses_all, pss))
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min))
            rots_all = np.vstack((rots_all, rotation))
        if split == 'train':
            self.poses_max = np.max(self.poses_max, axis=0) + mean_t[:2]
            self.poses_min = np.min(self.poses_min, axis=0) + mean_t[:2]
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
        poses_all_real = poses_all[:, :2] + mean_t[:2] - self.poses_min
        # divide the area into subregions
        if block_num != 1:
            for i in range(len(poses_all)):
                if int((poses_all_real[i, 0]) / block_size[0]) == num_grid_x and int(
                        poses_all_real[i, 1] / block_size[1]) == num_grid_y:
                    self.poses = np.vstack((self.poses, poses_all[i]))
                    self.pcs.append(pcs_all[i])
                    self.rots = np.vstack((self.rots, rots_all[i].reshape(1, 3, 3)))
        else:
            self.poses = poses_all
            self.pcs = pcs_all
            self.rots = rots_all

        self.split = split
        # self.augment = augment
        # if self.augment:
        #     print("=============use data augment=============")
        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        # print(scan_path)
        # with mask
        if self.train:
            ptcld = load_velodyne_binary_seg_feature32(scan_path).reshape(-1, 35)
            # ptcld = load_velodyne_binary(scan_path).reshape(4, -1).transpose()   # (N, 4)
        else:
            ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()  # (N, 4)
            ptcld[:, 2] = -1 * ptcld[:, 2]
        xyz = ptcld[:, :3]  # (N, 3)
        feature = ptcld[:, 3:]
        pose = self.poses[index]  # (6,)
        rot = self.rots[index]
        # ground truth
        gt = (rot @ xyz.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
  
        labels = np.concatenate((xyz, gt, feature), axis=1)
        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['labels'] = labels
        data_dict['pose'] = pose
        data_dict['rot'] = rot
        # data_dict['file_name'] = scan_path
        # data_dict['intensity'] = intensity

        return data_dict

    def __len__(self):
        return len(self.poses)
