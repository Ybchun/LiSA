import os
import numpy as np
import os.path as osp
import h5py
from torch.utils import data
from data.robotcar_sdk.python.velodyne import load_velodyne_binary, load_velodyne_binary_seg_feature96, load_velodyne_binary_seg_feature36
from util.pose_util import filter_overflow_nclt, interpolate_pose_nclt, so3_to_euler_nclt, process_poses

BASE_DIR = osp.dirname(osp.abspath(__file__))


class NCLT(data.Dataset):
    def __init__(self, data_path, split='train', real=False,):
        # directories
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'NCLT')

        # decide which sequences to use
        if split == 'train':
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        # 由于NCLT使用SLAM获取真值，故无需加入外参

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq )
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_' + str(real) + '.h5')
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_raw = []
                # 读入LiDAR时间戳，并从小到大排序
                vel = os.listdir(seq_dir + '/velodyne_left')
                for i in range(len(vel)):
                    ts_raw.append(int(vel[i][:-4]))
                ts_raw = sorted(ts_raw)
                # GT poses
                gt_filename = osp.join(seq_dir, 'groundtruth_' + seq + '.csv')
                ts[seq] = filter_overflow_nclt(gt_filename, ts_raw)
                p = interpolate_pose_nclt(gt_filename, ts[seq])  # (n, 6)
                p = so3_to_euler_nclt(p)  # (n, 4, 4)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)

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

            # self.pcs.extend(
            #     [osp.join(seq_dir, 'sphere_velodyne_left_feature32', '{:d}.bin'.format(t)) for t in ts[seq]])
            self.pcs.extend(
                [osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

            # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'NCLT_pose_stats.txt')
        if split == 'train':
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        # with mask
        # ptcld = load_velodyne_binary_seg_feature36(scan_path)  # (N, 4)
        ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        xyz = ptcld[:, :3]  # (N, 3)

        # mask = ptcld[:, 3].reshape(-1, 1)  # mask label
        # mask = ptcld[:, 3:35]

        pose = self.poses[index]  # (6,)
        rot = self.rots[index]
        # ground truth
        gt = (rot @ xyz.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        # labels = np.concatenate((xyz, gt, mask), axis=1)
        labels = np.concatenate((xyz, gt), axis=1)
        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['labels'] = labels
        data_dict['pose'] = pose
        data_dict['rot'] = rot

        return data_dict

    def __len__(self):
        return len(self.poses)