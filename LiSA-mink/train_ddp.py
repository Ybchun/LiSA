# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib
import matplotlib.pyplot as plt
from utils.avgMeter import AverageMeter
# The quality-enhanced Oxford dataset
from data.Enhanced_OxfordVelodyne_datagenerator import RobotCar

from data.NCLTVelodyne_datagenerator import NCLT
from models.model import LiSA
from models.loss import CriterionCoordinate, Seg_CriterionCoordinate, Seg_DDPM_CriterionCoordinate
# from models.loss import Neighborhood_Distance
from data.base_loader import CollationFunctionFactory
from utils.pose_util import val_translation, val_rotation, qexp, estimate_poses
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp
from tqdm import tqdm
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None,
                    help='Model checkpoint path [default: None]')
parser.add_argument('--batch_size', type=int, default=40,
                    help='Batch Size during training')
parser.add_argument('--val_batch_size', type=int, default=20,
                    help='Batch Size during validating [default: 30]')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='Epoch to run [default: 41]')
parser.add_argument('--init_learning_rate', type=float, default=0.01,
                    help='Initial learning rate [default: 0.005]')
parser.add_argument("--decay_step", type=float, default=500,
                    help="decay step for learning rate, default:1200")
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='log_ddp_NCLT',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/Disk24_2/data',
                    help='Our Dataset Folder')
parser.add_argument('--dataset', default='NCLT',
                    help='Oxford or NCLT')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num workers for dataloader, default:4')
parser.add_argument('--voxel_size', type=float, default=0.2,
                    help='voxel size [default: Oxford 0.2, NCLT 0.25]')

# arguments for ddp training
parser.add_argument('--use_ddp', type=int, default=1,
                    help='using ddp to train model')
parser.add_argument('--local_rank', type=int, default=0,
                    help="If using ddp to train")

"""
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.34 --master_port 29503 train_ddp.py
"""

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

FLAGS = parser.parse_args()
args = vars(FLAGS)
for (k, v) in args.items():
    print('%s: %s' % (str(k), str(v)))
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
TOTAL_ITERATIONS = 0

if FLAGS.use_ddp > 0:
    print('local rank', FLAGS.local_rank)
    torch.cuda.set_device(FLAGS.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    device = torch.device(f'cuda:{FLAGS.local_rank}')
else:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("Don't use ddp to train....")

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # print(out_str)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

train_kwargs = dict(data_path=FLAGS.dataset_folder,
                            train=True,
                            valid=False,
                            voxel_size=FLAGS.voxel_size)

valid_kwargs = dict(data_path=FLAGS.dataset_folder,
                    train=False,
                    valid=True,
                    voxel_size=FLAGS.voxel_size)

pose_stats_file = os.path.join(FLAGS.dataset_folder, FLAGS.dataset, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)

if FLAGS.dataset == 'Oxford':
    train_set = RobotCar(**train_kwargs)
    val_set = RobotCar(**valid_kwargs)
elif FLAGS.dataset == 'NCLT':
    train_set = NCLT(**train_kwargs)
    val_set = NCLT(**valid_kwargs)
else:
    raise ValueError("dataset error!")
collation_fn = CollationFunctionFactory(collation_type='collate_pair')

class Trainer:
    def __init__(self):
        global TOTAL_ITERATIONS
        self.train_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
        self.val_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))

        self.get_tra_val_dataloader(train_set, val_set)

        self.net = LiSA()

        if FLAGS.use_ddp > 0:
            self.net.to(device=FLAGS.local_rank)
        else:
            self.net = self.net.to(device=device)

        self.criterion = Seg_CriterionCoordinate()
        

        self.optimizer = torch.optim.Adam(self.net.parameters(), FLAGS.init_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, FLAGS.decay_step, gamma=0.95)

        # Load module
        self.highest_sp_val_acc = 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda:{}'.format(FLAGS.local_rank))
            self.net.load_state_dict(checkpoint['state_dict'],
                                     strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']

        if not FLAGS.use_ddp:
            self.net.to(device)
        else:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[FLAGS.local_rank])
            self.net_single = self.net.module

        LOG_FOUT.write("\n")
        LOG_FOUT.flush()


    def train_one_epoch(self):
        self.net.train()
        losses = AverageMeter()
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        for batch_idx, batch_data in enumerate(tqdm_loader):
            # batch_data = self.send_data2GPU(batch_data)
            features = batch_data['sinput_F'].to(device, dtype=torch.float32)
            coordinates = batch_data['sinput_C'].to(device)
            pcs_tensor = ME.SparseTensor(features, coordinates)
            # label
            # 前三维是原始点云点，后三维是gt点云点
            features_s8 = batch_data['sinput_s8_F'].to(device, dtype=torch.float32)
            coordinates_s8 = batch_data['sinput_s8_C'].to(device)
            pcs_tensor_s8 = ME.SparseTensor(features_s8, coordinates_s8)
            # gt: t
            pose_gt = batch_data['pose'].to(device, dtype=torch.float32)
            batch_size = pose_gt.size(0)

            self.optimizer.zero_grad()

            # pred_shift, ddim_loss, kd_loss = self.net(pcs_tensor, pcs_tensor_s8)
            pred_shift, pred_seg = self.net(pcs_tensor)
            # pred_shift = self.net(pcs_tensor)
            index_list = [0]  # 用于存放索引
            for i in range(batch_size):
                # 取出预测的每个batch中的坐标点
                batch_pred_pcs_tensor = pred_shift.coordinates_at(i).float()
                index_list.append(index_list[i] + len(batch_pred_pcs_tensor))

            ground_truth = pcs_tensor_s8.features_at_coordinates(
                torch.cat((pred_shift.C[:, 0].view(-1, 1), pred_shift.C[:, 1:] / 8), axis=1).float())

            sup_point = ground_truth[:, :3]
            gt_sup_point = ground_truth[:, 3:6]
            # mask: 仅用于计算loss
            # mask = ground_truth[:, 6].view(-1, 1)
            gt_seg = ground_truth[:, 6:]

            pred_point = sup_point + pred_shift.F
            # loss = self.criterion(pred_point, gt_sup_point, ddim_loss, kd_loss)
            loss = self.criterion(pred_point, gt_sup_point, pred_seg.F, gt_seg)
            # loss = self.criterion(pred_point, gt_sup_point)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), FLAGS.batch_size)

            log_string('Loss: %f' % loss)
            self.train_writer.add_scalar('Loss', loss.cpu().item(), TOTAL_ITERATIONS)
            # if (FLAGS.local_rank == 0 and FLAGS.use_ddp) or not FLAGS.use_ddp:
            self.train_writer.add_scalar(
                'train/loss', loss.item(), self.cur_epoch * len(self.train_loader) + batch_idx)


    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            log_string('**** EPOCH %03d ****' % epoch)
            if FLAGS.use_ddp:
                self.train_sampler.set_epoch(epoch)
            # sys.stdout.flush()
            self.train_one_epoch()
            self.scheduler.step()
            # if epoch % 5 ==0 or epoch>=10:
            if (FLAGS.local_rank == 0 and FLAGS.use_ddp and epoch > 30) or not FLAGS.use_ddp:
                log_string('**** EVAL EPOCH %03d ****' % epoch)
                checkpoint_file = os.path.join(FLAGS.log_dir, 'checkpoint.tar')
                self.save_checkpoint(checkpoint_file)

                if self.cur_epoch % 1 == 0:
                    checkpoint_file = os.path.join(
                        FLAGS.log_dir, 'checkpoint_epoch_{}.tar'.format(self.cur_epoch))
                    self.save_checkpoint(checkpoint_file)

    @staticmethod
    def send_data2GPU(batch_data):
        for key in batch_data:  # Target data to gpu
            batch_data[key] = batch_data[key].to(device=device, dtype=torch.float32)
        return batch_data

    def save_checkpoint(self, fname):
        save_dict = {
            # after training one epoch, the start_epoch should be epoch+1
            'epoch': self.cur_epoch+1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)

    def get_tra_val_dataloader(self, train_dataset, val_dataset):
        if not FLAGS.use_ddp:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=FLAGS.batch_size,
                shuffle=True,
                num_workers=FLAGS.num_workers,
                worker_init_fn=my_worker_init_fn,
                collate_fn=collation_fn,
                pin_memory=True
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=FLAGS.val_batch_size,
                num_workers=FLAGS.num_workers,
                worker_init_fn=my_worker_init_fn,
                collate_fn=collation_fn,
                pin_memory=True
            )
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=FLAGS.batch_size,
                num_workers=FLAGS.num_workers,
                worker_init_fn=my_worker_init_fn,
                collate_fn=collation_fn,
                pin_memory=True,
                sampler=self.train_sampler
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=FLAGS.val_batch_size,
                num_workers=FLAGS.num_workers,
                worker_init_fn=my_worker_init_fn,
                collate_fn=collation_fn,
                pin_memory=True,
                sampler=valid_sampler
            )

# @email_on_exception(['1050506643@qq.com'])
def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()