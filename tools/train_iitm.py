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

from dataset import *

torch.backends.cudnn.benchmark = True

import sys
sys.path.append('..')


from src.models import *

#from env_utils import *

import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

import wandb
from patchify import unpatchify
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def get_parser():
    parser = argparse.ArgumentParser(description='SIRFormer')
#    parser.add_argument('-cfg', '--cfg', '--config', default='./', help='config path')
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None, help='load model')
    parser.add_argument('--savemodel', default=None, help='save model')
#    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
#    parser.add_argument('--devices', '-d', type=str, default=None)
    parser.add_argument('--lr_scale', type=int, default=40, metavar='S', help='lr scale')

    parser.add_argument('--trainset_dir', type=str, default='../../iPASSR_trainset')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument("--scale_factor", type=int, default=2)

    parser.add_argument('--btrain', '-btrain', type=int, default=24)
    parser.add_argument('--start_epoch', type=int, default=None)
    
    parser.add_argument('--upscale_factor', default=2, type=int,
                        help="SR upscale factor")
    
    parser.add_argument('--exposure', type=str, default='1_25')

    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--val_event_freq', type=int, default=10)
    
    # for positional encoding
#    parser.add_argument('--position_encoding', default='sine1d_rel', type=str, choices=('sine1d_rel', 'none'),
#                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--channel_dim', default=32, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--patch_size', default=(128, 256), type=int,
                        help="patch size for transformer")


    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()


    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = get_parser()
    set_seed(args.seed)

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))

    # ----------------model    
    model_transformer = SIRFormer(args)
    # ----------------optimizer
    optimizer = optim.Adam(model_transformer.parameters(), lr=0.0008, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 100, 120], gamma=args.gamma)
    # ----------------criterion
    L1 = nn.L1Loss().cuda()

    model_transformer = torch.nn.DataParallel(model_transformer).cuda()

    #------------------- Data Loader -----------------------
    train_set = TrainSetLoader(exposure=args.exposure, patch_size=args.patch_size)

    train_sampler = None
    TrainImgLoader = torch.utils.data.DataLoader(
        dataset=train_set, 
        batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=True,
        sampler=train_sampler)

    val_set = ValSetLoader(exposure=args.exposure, patch_size=args.patch_size)

    print("Len of Train set:", len(train_set))
    print("Len of Val set:", len(val_set))


    #------------------ Logger -------------------------------------
    wandb.init(project='iitm-sirformer', config={"exposure":args.exposure})
    # logger = exp.logger
    # # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model_transformer.parameters()])))
    # writer = exp.writer



    # ------------------------ Resume ------------------------------
    if args.start_epoch is None:
        args.start_epoch = 1

    # ------------------------ Training ------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        total_train_loss = 0

        for batch_idx, samples in enumerate(TrainImgLoader):
            start_time = time.time()
            (img_left_hr, img_right_hr, img_left_lr, img_right_lr) = samples["wl_l"], samples["wl_r"], samples["ll_l"], samples["ll_r"]
            losses = train(model_transformer, L1, optimizer, img_left_lr, img_right_lr, img_left_hr, img_right_hr)
            loss = losses.pop('loss')

            wandb.log({"loss": loss, "l1_loss":losses.pop('loss_l1'), "embedding_loss":losses.pop('loss_embedding')})

            # print('%s: %s' % (args.savemodel.strip('/').split('/')[-1], args.devices))
            # print('Epoch %d Iter %d/%d training loss = %.3f , time = %.2f; Epoch time: %.3fs, Left time: %.3fs, lr: %.6f' % (
            #     epoch,
            #     batch_idx, len(TrainImgLoader), loss, time.time() - start_time, (time.time() - start_time) * len(TrainImgLoader),
            #     (time.time() - start_time) * (len(TrainImgLoader) * (args.epochs - epoch) - batch_idx), optimizer.param_groups[0]["lr"]) )
            # print('losses: {}'.format(list(losses.items())))
            # for lk, lv in losses.items():
            #     writer.add_scalar(lk, lv, epoch * len(TrainImgLoader) + batch_idx)
            total_train_loss += loss


        scheduler.step()

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        savefilename = '/home/adithyal/Stereo_LLE/Transformer/ckpts/' + args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model_transformer.module.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
            'optimizer': optimizer.state_dict()
        }, savefilename)
        print('Snapshot {} epoch in {}'.format(epoch, args.savemodel))

        if epoch%args.val_freq==0:
            total_metrics = {"Val_PSNR":0, "Val_SSIM":0, "Val_Loss":0}
            for sample in val_set:
                metrics = evaluate(model_transformer, L1, sample)
                total_metrics['Val_PSNR'] += metrics['PSNR']
                total_metrics['Val_SSIM'] += metrics['SSIM']
                total_metrics['Val_Loss'] += metrics['Loss']
            for key in total_metrics:
                total_metrics[key] = total_metrics[key]/len(val_set)

            wandb.log(total_metrics)

        if epoch%args.val_event_freq==0:
            for i, sample in enumerate(val_set):
                if i%10==0:
                    ll_l_img, wl_l_img, left_img_pred, ll_r_img, wl_r_img, right_img_pred = pred(model_transformer, sample)
                    ll_l_img = wandb.Image(ll_l_img, caption= f"L input")
                    wl_l_img = wandb.Image(wl_l_img, caption= f"L groundtruth")
                    left_img_pred = wandb.Image(left_img_pred, caption= f"L pred")
                    ll_r_img = wandb.Image(ll_r_img, caption= f"R input")
                    wl_r_img = wandb.Image(wl_r_img, caption= f"R groundtruth")
                    right_img_pred = wandb.Image(right_img_pred, caption= f"R pred")
                    wandb.log({f"val {i}": [ll_l_img, wl_l_img, left_img_pred, ll_r_img, wl_r_img, right_img_pred]})



def pred(model_transformer, sample):
    model_transformer.eval()
    ll_l_patches = sample['ll_l'].cuda()
    ll_r_patches = sample['ll_r'].cuda()
    wl_l_patches = sample['wl_l'].cuda()
    wl_r_patches = sample['wl_r'].cuda()

    ll_l_img = unpatchify(ll_l_patches.permute(0,2,3,1).cpu().numpy().reshape(8,8,1, 128, 256, 3), (1024, 2048, 3))
    ll_r_img = unpatchify(ll_r_patches.permute(0,2,3,1).cpu().numpy().reshape(8,8,1, 128, 256, 3), (1024, 2048, 3))
    
    wl_l_img = unpatchify(wl_l_patches.permute(0,2,3,1).cpu().numpy().reshape(8,8,1, 128, 256, 3), (1024, 2048, 3))
    wl_r_img = unpatchify(wl_r_patches.permute(0,2,3,1).cpu().numpy().reshape(8,8,1, 128, 256, 3), (1024, 2048, 3))

    left_img_pred = torch.zeros_like(wl_l_patches, device='cpu')
    right_img_pred = torch.zeros_like(wl_r_patches, device='cpu')

    num_patches = ll_l_patches.shape[0]
    with torch.no_grad():
        for i in range(num_patches):
            l_pred, r_pred = model_transformer(ll_l_patches[i].unsqueeze(dim=0), ll_r_patches[i].unsqueeze(dim=0))
            left_img_pred[i] = l_pred
            right_img_pred[i] = r_pred
    
    left_img_pred = unpatchify(left_img_pred.permute(0,2,3,1).numpy().reshape(8,8,1,128,256,3), (1024,2048,3))
    right_img_pred = unpatchify(right_img_pred.permute(0,2,3,1).numpy().reshape(8,8,1,128,256,3), (1024,2048,3))
    
    return ll_l_img, wl_l_img, left_img_pred, ll_r_img, wl_r_img, right_img_pred


def evaluate(model_transformer, L1, sample):
    model_transformer.eval()
    ll_l_patches = sample['ll_l'].cuda()
    ll_r_patches = sample['ll_r'].cuda()
    wl_l_patches = sample['wl_l'].cuda()
    wl_r_patches = sample['wl_r'].cuda()

    wl_l_img = unpatchify(wl_l_patches.permute(0,2,3,1).cpu().numpy().reshape(8,8,1, 128, 256, 3), (1024, 2048, 3))
    wl_r_img = unpatchify(wl_r_patches.permute(0,2,3,1).cpu().numpy().reshape(8,8,1, 128, 256, 3), (1024, 2048, 3))

    left_img = torch.zeros_like(wl_l_patches, device='cpu')
    right_img = torch.zeros_like(wl_r_patches, device='cpu')

    num_patches = ll_l_patches.shape[0]
    total_loss = 0
    with torch.no_grad():    
        for i in range(num_patches):
            l_pred, r_pred = model_transformer(ll_l_patches[i].unsqueeze(dim=0), ll_r_patches[i].unsqueeze(dim=0))
            total_loss += L1(l_pred, wl_l_patches[i]) + L1(r_pred, wl_r_patches[i])
            left_img[i] = l_pred
            right_img[i] = r_pred

    left_img = unpatchify(left_img.permute(0,2,3,1).numpy().reshape(8,8,1,128,256,3), (1024,2048,3))
    right_img = unpatchify(right_img.permute(0,2,3,1).numpy().reshape(8,8,1,128,256,3), (1024,2048,3))

    psnr_l = psnr(left_img, wl_l_img, data_range = left_img.max() - left_img.min())
    psnr_r = psnr(right_img, wl_r_img, data_range = right_img.max() - right_img.min())

    ssim_l = ssim(left_img, wl_l_img, channel_axis=2, data_range = left_img.max() - left_img.min())
    ssim_r = ssim(right_img, wl_r_img, channel_axis=2, data_range = right_img.max() - right_img.min())

    return {"PSNR": (psnr_l+psnr_r)/2, "SSIM":(ssim_l+ssim_r)/2, "Loss": total_loss/num_patches}
  


def train(model_transformer, L1, optimizer, imgL_lr, imgR_lr, img_left_hr, img_right_hr,):
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

