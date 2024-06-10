# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import torch
import matplotlib
import torch.nn as nn

matplotlib.use('Agg')
# Oxford
from data.oxford import Oxford
# QEOxford
from data.qeoxford import QEOxford
# NCLT
from data.nclt import NCLT
from util import config
from model.lisa import SGLoc, LiSA, LiSA_ddpm
from model.loss import L1_CriterionCoordinate, CriterionCoordinate, DDPM_CriterionCoordinate
from data.dataset import sgloc_data, collate_fn_default
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.34 --master_port 29503 train_ddp.py

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Localization')
    parser.add_argument('--config', type=str, default='config/NCLT/lisa_asc.yaml',
                        help='config file')
    parser.add_argument('--local_rank', type=int, default=0,
                        help="If using ddp to train")
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.update(vars(args))
    return cfg

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # print(out_str)


def main(args):
    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = 0

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f'cuda:{args.local_rank}')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.enabled = True
        cudnn.deterministic = True

    for (k, v) in args.items():
        print('%s: %s' % (str(k), str(v)))

    if args.data_name == 'Oxford':
        train_pt_dataset = Oxford(args.data_root,
                                    split='train')
    elif args.data_name == 'QEOxford':
        train_pt_dataset = QEOxford(args.data_root,
                                  split='train')
    elif args.data_name == 'NCLT':
        train_pt_dataset = NCLT(args.data_root,
                                  split='train')
    else:
        print('Only supprot Oxford, QEOxford and NCLT')

    dataset = sgloc_data(train_pt_dataset, args)
    tran_sampler = DistributedSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               collate_fn=collate_fn_default,
                                               sampler=tran_sampler
                                               )

    pose_stats_file = os.path.join(args.data_root, args.data_name[-6:], args.data_name + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)

    train_writer = SummaryWriter(os.path.join(args.save_path, 'train'))

    model = SGLoc(args)
    # model = LiSA(args)

    model = model.to(device=args.local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)
    # loss = L1_CriterionCoordinate()
    loss = CriterionCoordinate()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=0.95)


    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    for epoch in range(args.epochs):
        log_string('**** EPOCH %03d ****' % epoch)
        sys.stdout.flush()
        train_one_epoch(model, train_loader, tran_sampler, scheduler, epoch, train_writer, loss, args, device)


def train_one_epoch(model, train_loader, tran_sampler, scheduler, epoch, train_writer, loss, args, device):
    global TOTAL_ITERATIONS
    model.train()
    tran_sampler.set_epoch(epoch)
    tqdm_loader = tqdm(train_loader, total=len(train_loader))
    for _, input_dict in enumerate(tqdm_loader):
        TOTAL_ITERATIONS += 1
        input_dict['points'] = input_dict['points'].to(device, dtype=torch.float32)
        input_dict['batch_idx'] = input_dict['batch_idx'].to(device, dtype=torch.int8)
        input_dict['labels'] = input_dict['labels'].to(device, dtype=torch.float32)
        scheduler.optimizer.zero_grad()

        # pred_shift, pred_seg_feature, labels, _ = model(input_dict)
        pred_shift, labels, _ = model(input_dict)

        sup_point = labels[:, :3]
        gt_sup_point = labels[:, 3:6]
        # seg_feature = labels[:, 6:]

        pred_point = sup_point + pred_shift

        # train_loss = loss(pred_point, gt_sup_point, pred_seg_feature, seg_feature)
        train_loss = loss(pred_point, gt_sup_point)

        train_loss.backward(train_loss)
        scheduler.optimizer.step()
        scheduler.step()
        log_string('Loss: %f' % train_loss)
        train_writer.add_scalar('Loss', train_loss.cpu().item(), TOTAL_ITERATIONS)

    if epoch % 1 == 0 and torch.distributed.get_rank() == 0:
        model_to_save = model.module
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            args.save_path + 'checkpoint_epoch{}.tar'.format(epoch))
        print("Model Saved As " + 'checkpoint_epoch{}.tar'.format(epoch))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import gc

    gc.collect()
    args = get_parser()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    LOG_FOUT = open(os.path.join(args.save_path, 'log_sgloc.txt'), 'w')
    LOG_FOUT.write(str(args) + '\n')
    # 占用2个核心
    torch.set_num_threads(4)
    main(args)