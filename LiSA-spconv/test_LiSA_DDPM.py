# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import time
import torch

import matplotlib
# Oxford
from data.oxford import Oxford
# QEOxford
from data.qeoxford import QEOxford
# NCLT
from data.nclt import NCLT
from util import config
from model.lisa import LiSA, LiSA_ddpm
from model.sc2pcr import Matcher
from data.dataset import sgloc_data, collate_fn_default
from util.pose_util import val_translation, val_rotation, qexp
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# 过滤指定类型的警告信息
warnings.filterwarnings("ignore", category=UserWarning)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Localization')
    parser.add_argument('--config', type=str, default='config/QEOxford/lisa.yaml',
                        help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # print(out_str)


def main(args):
    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = 0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    for (k, v) in args.items():
        print('%s: %s' % (str(k), str(v)))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")

    if args.data_name == 'Oxford':
        test_pt_dataset = Oxford(args.data_root,
                                 split='test')
    elif args.data_name == 'QEOxford':
        test_pt_dataset = QEOxford(args.data_root,
                                   split='test')
    elif args.data_name == 'NCLT':
        test_pt_dataset = NCLT(args.data_root,
                               split='test')
    else:
        print('Only supprot Oxford, QEOxford and NCLT')

    lenset = len(test_pt_dataset)

    test_loader = torch.utils.data.DataLoader(sgloc_data(test_pt_dataset, args),
                                               batch_size=args.batch_size_val,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               collate_fn=collate_fn_default
                                               )
    test_writer = SummaryWriter(os.path.join(args.save_path, 'test'))

    model = LiSA_ddpm(args)
    model = model.to(device)
    ransac = Matcher(inlier_threshold=args.threshold)

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    for epoch in range(1, 2, 1):
        log_string('**** EPOCH %03d ****' % epoch)
        resume_filename = args.resume + str(epoch) + '.tar'
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename, 'cuda:0')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
        sys.stdout.flush()
        valid_one_epoch(model, ransac, test_loader, test_writer, lenset, device, args.threshold)
        torch.cuda.empty_cache()

def valid_one_epoch(model, ransac, test_loader, test_writer, lenset, device, threshold):
    gt_translation = np.zeros((lenset, 3))
    pred_translation = np.zeros((lenset, 3))
    gt_rotation = np.zeros((lenset, 4))
    pred_rotation = np.zeros((lenset, 4))

    error_t = np.zeros(lenset)
    error_txy = np.zeros(lenset)
    error_q = np.zeros(lenset)

    time_results_network = np.zeros(lenset)
    time_results_ransac = np.zeros(lenset)

    tqdm_loader = tqdm(test_loader, total=len(test_loader))
    for step, input_dict in enumerate(tqdm_loader):
        val_pose = input_dict['pose'].reshape(-1, 6)
        batch_size = val_pose.size(0)
        pred_t = np.zeros((batch_size, 3))
        pred_q = np.zeros((batch_size, 4))
        index_list = [0] # 用于存放索引
        start_idx = step * args.batch_size_val
        end_idx = min((step + 1) * args.batch_size_val, lenset)
        # 在得到gt时，仅减去了均值
        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() + pose_m
        gt_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()])
        # 需要GPU计算的数据
        input_dict['points'] = input_dict['points'].to(device, dtype=torch.float32)
        input_dict['batch_idx'] = input_dict['batch_idx'].to(device, dtype=torch.int8)
        input_dict['labels'] = input_dict['labels'].to(device, dtype=torch.float32)

        # inference model and time cost,仅能在batch size 为1的时准确
        start = time.time()
        pred_shift, labels, _, __, coors = run_model(model, input_dict, True)
        end = time.time()
        cost_time = (end - start) / batch_size
        time_results_network[start_idx:end_idx] = [cost_time for _ in range(batch_size)]

        sup_point = labels[:, :3]
        pred_point = sup_point + pred_shift
        # 现在情况：输出点是输入点的子集
        for i in range(batch_size):
            # 取出预测的每个batch中的坐标点
            index_list.append(index_list[i] + torch.sum(coors[:, 0]==i))
            # print(torch.sum(coors[:, 0]==i))
        ori_point = sup_point

        start = time.time()
        for i in range(batch_size):
            batch_pred_t, batch_pred_q = ransac.estimator(
                        ori_point[index_list[i]:index_list[i + 1], :].unsqueeze(0) \
                        , pred_point[index_list[i]:index_list[i + 1], :].unsqueeze(0))
            pred_t[i, :] = batch_pred_t
            pred_q[i, :] = batch_pred_q
        end = time.time()
        cost_time = (end - start) / batch_size
        time_results_ransac[start_idx:end_idx] = [cost_time for _ in range(batch_size)]

        pred_translation[start_idx:end_idx, :] = pred_t + pose_m
        pred_rotation[start_idx:end_idx, :] = pred_q

        error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :],
                                                     gt_translation[start_idx:end_idx, :])])
        error_txy[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :2],
                                                     gt_translation[start_idx:end_idx, :2])])

        error_q[start_idx:end_idx] = np.asarray([val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :],
                                                                                    gt_rotation[start_idx:end_idx, :])])

        log_string('MeanXYZTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanXYTE(m): %f' % np.mean(error_txy[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))
        log_string('NetCostTime(s): %f' % np.mean(time_results_network[start_idx:end_idx], axis=0))
        log_string('RANSACCostTime(s): %f' % np.mean(time_results_ransac[start_idx:end_idx], axis=0))

    mean_ATE = np.mean(error_t)
    mean_xyATE = np.mean(error_txy)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_xyATE = np.median(error_txy)
    median_ARE = np.median(error_q)
    mean_time_network = np.mean(time_results_network)
    mean_time_ransac = np.mean(time_results_ransac)

    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean XY Position Error(m): %f' % mean_xyATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median XY Position Error(m): %f' % median_xyATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)
    log_string('Mean Network Cost Time(s): %f' % mean_time_network)
    log_string('Mean Ransac Cost Time(s): %f' % mean_time_ransac)
    test_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    test_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)

    # save error
    error_t_filename = osp.join(args.save_path, 'error_t.txt')
    error_q_filename = osp.join(args.save_path, 'error_q.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')

    # trajectory
    fig = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=1, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=1, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(args.save_path), '{:s}.png'.format('trajectory_t1' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # translation_distribution
    fig = plt.figure()
    t_num = np.arange(len(error_t))
    plt.scatter(t_num, error_t, s=1, c='red')
    plt.xlabel('Data Num')
    plt.ylabel('Error (m)')
    image_filename = os.path.join(os.path.expanduser(args.save_path), '{:s}.png'.format('distribution_t_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # rotation_distribution
    fig = plt.figure()
    q_num = np.arange(len(error_q))
    plt.scatter(q_num, error_q, s=1, c='blue')
    plt.xlabel('Data Num')
    plt.ylabel('Error (degree)')
    image_filename = os.path.join(os.path.expanduser(args.save_path), '{:s}.png'.format('distribution_q_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(args.save_path, 'error_t.txt')
    error_q_filename = osp.join(args.save_path, 'error_q.txt')
    pred_q_filename = osp.join(args.save_path, 'pred_q.txt')
    pred_t_filename = osp.join(args.save_path, 'pred_t.txt')
    gt_t_filename = osp.join(args.save_path, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')


def run_model(model, x1, validate=False):
    if not validate:
        model.train()
        return model(x1)
    else:
        with torch.no_grad():
            model.eval()
            return model(x1)


if __name__ == "__main__":
    import gc
    gc.collect()
    args = get_parser()
    LOG_FOUT = open(os.path.join(args.save_path, 'log_test_LiSA_ddpm.txt'), 'w')
    LOG_FOUT.write(str(args) + '\n')
    # 读入训练时生成的统计数据
    pose_stats_file = os.path.join(args.data_root, args.data_name[-6:], args.data_name + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)
    # 占用2个核心
    torch.set_num_threads(2)
    main(args)