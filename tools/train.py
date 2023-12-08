from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from utils import *

torch.backends.cudnn.benchmark = True

import sys
sys.path.append('..')


from src.models import *

from env_utils import *

import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description='SIRFormer')
    parser.add_argument('-cfg', '--cfg', '--config', default='./', help='config path')
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None, help='load model')
    parser.add_argument('--savemodel', default=None, help='save model')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--devices', '-d', type=str, default=None)
    parser.add_argument('--lr_scale', type=int, default=40, metavar='S', help='lr scale')

    parser.add_argument('--trainset_dir', type=str, default='../../iPASSR_trainset')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument("--scale_factor", type=int, default=2)

    parser.add_argument('--btrain', '-btrain', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    
    parser.add_argument('--upscale_factor', default=2, type=int,
                        help="SR upscale factor")
    
    # for positional encoding
    parser.add_argument('--position_encoding', default='sine1d_rel', type=str, choices=('sine1d_rel', 'none'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--channel_dim', default=32, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--patch_size', default=(30, 90), type=int,
                        help="patch size for transformer")


    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()


    return args


def main():

    args = get_parser()

    global cfg
    exp = Experimenter(args.savemodel, cfg_path=args.cfg)
    cfg = exp.config

    reset_seed(args.seed)

    cfg.debug = args.debug
    cfg.warmup = getattr(cfg, 'warmup', True) if not args.debug else False

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))

    main_worker(0, ngpus_per_node, args, cfg, exp)

def main_worker(gpu, ngpus_per_node, args, cfg, exp):
    # ----------------model    
    model_transformer = SIRFormer(args)
    # ----------------optimizer
    optimizer = optim.Adam(model_transformer.parameters(), lr=0.0008, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 100, 120], gamma=args.gamma)
    # ----------------criterion
    L1 = nn.L1Loss().cuda()

    model_transformer = torch.nn.DataParallel(model_transformer).cuda()

    #------------------- Data Loader -----------------------
    train_set = TrainSetLoader(args)

    train_sampler = None
    TrainImgLoader = torch.utils.data.DataLoader(
        dataset=train_set, 
        batch_size=args.btrain, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True,
        sampler=train_sampler)

    args.max_warmup_step = min(len(TrainImgLoader), 500)

    #------------------ Logger -------------------------------------

    logger = exp.logger
    # logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model_transformer.parameters()])))
    writer = exp.writer

    # ------------------------ Resume ------------------------------
    if args.start_epoch is None:
        args.start_epoch = 1

    # ------------------------ Training ------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        total_train_loss = 0

        for batch_idx, (img_left_hr, img_right_hr, img_left_lr, img_right_lr) in enumerate(TrainImgLoader):
            start_time = time.time()

            losses = train(model_transformer, L1, cfg, args, optimizer, img_left_lr, img_right_lr, img_left_hr, img_right_hr)
            loss = losses.pop('loss')

            logger.info('%s: %s' % (args.savemodel.strip('/').split('/')[-1], args.devices))
            logger.info('Epoch %d Iter %d/%d training loss = %.3f , time = %.2f; Epoch time: %.3fs, Left time: %.3fs, lr: %.6f' % (
                epoch,
                batch_idx, len(TrainImgLoader), loss, time.time() - start_time, (time.time() - start_time) * len(TrainImgLoader),
                (time.time() - start_time) * (len(TrainImgLoader) * (args.epochs - epoch) - batch_idx), optimizer.param_groups[0]["lr"]) )
            logger.info('losses: {}'.format(list(losses.items())))
            for lk, lv in losses.items():
                writer.add_scalar(lk, lv, epoch * len(TrainImgLoader) + batch_idx)
            total_train_loss += loss


        scheduler.step()

        logger.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model_transformer.module.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
            'optimizer': optimizer.state_dict()
        }, savefilename)
        logger.info('Snapshot {} epoch in {}'.format(epoch, args.savemodel))


def train(model_transformer, L1, cfg, args, optimizer, imgL_lr, imgR_lr, img_left_hr, img_right_hr,):
    # get_loss= My_loss(10, 5, 2, 3)
    model_transformer.train()
    imgL_lr = Variable(torch.FloatTensor(imgL_lr))
    imgR_lr = Variable(torch.FloatTensor(imgR_lr))
    imgL_hr = Variable(torch.FloatTensor(img_left_hr))
    imgR_hr = Variable(torch.FloatTensor(img_right_hr))

    imgL_lr, imgR_lr, imgL_hr, imgR_hr = imgL_lr.cuda(), imgR_lr.cuda(), imgL_hr.cuda(), imgR_hr.cuda()

    losses = dict()
    
    left_sr, right_sr = model_transformer(imgL_lr, imgR_lr)
    
    # for embedding loss
    # step1: whiten(option)
    # step2: compute map
    b, c, h, w = left_sr.shape
    score = torch.bmm(left_sr.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                      right_sr.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
    M_right_to_left = torch.nn.Softmax(-1)(score)
    M_left_to_right = torch.nn.Softmax(-1)(score.permute(0, 2, 1))
    # step3: wrapped
    left_sr_wrapped = torch.bmm(M_right_to_left, right_sr.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
                                ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

    right_sr_wrapped = torch.bmm(M_left_to_right, left_sr.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
                                 ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

    # SR_Loss
    loss_sr = L1(left_sr, imgL_hr) + L1(right_sr, imgR_hr)
    # Embedding_Loss
    loss_embedding = L1(left_sr, left_sr_wrapped) + L1(right_sr, right_sr_wrapped)

    loss = loss_sr + 0.01 * loss_embedding
    losses.update(loss=loss)
    losses.update(loss_l1=loss_sr)
    losses.update(loss_embedding=loss_embedding)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    reduced_losses = {k: v.item() for k, v in losses.items()}

    return reduced_losses

if __name__ == '__main__':
    main()

