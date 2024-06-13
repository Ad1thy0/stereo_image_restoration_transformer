#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
#from torch._six import container_abcs
import collections.abc as container_abcs

from .model import *
from .rcan import RCAB, default_conv, RCAN
import torch.nn.functional as F


layer_idx = 0

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class ConvDW3x3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=1,
            groups=dim)

    def forward(self, x):
        output = self.conv(x)
        return output + x

class Transformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, img_size = (30, 90), hidden_dim: int = 64, nhead_1: int = 4, nhead_2=2, nhead_3 = 4, num_attn_layers: int = 6, depth = 1, win_size = 8):
        super().__init__()

        self.img_size = to_2tuple(img_size)


        self.depthwise_conv_for_self1 = ConvDW3x3(dim=hidden_dim)
        self.depthwise_conv_for_self2 = ConvDW3x3(dim=hidden_dim)
        self.depthwise_conv_for_cross1 = ConvDW3x3(dim=hidden_dim)
        self.depthwise_conv_for_fusion = ConvDW3x3(dim=hidden_dim)

        # self-attention-module

        self.self_attn_layers1 =  nn.ModuleList([
            SelfAttnBlock(dim=hidden_dim, input_resolution=img_size, num_heads=nhead_1, win_size=win_size,
                                 shift_size=0 if (i % 2 == 0) else win_size // 2, token_mlp='ffn')
            for i in range(depth)])
        self.self_attn_layers2 =  nn.ModuleList([
            SelfAttnBlock(dim=hidden_dim, input_resolution=img_size, num_heads=nhead_1, win_size=win_size,
                                 shift_size=0 if (i % 2 == 0) else win_size // 2, token_mlp='ffn')
            for i in range(depth)])

        self.SAT = SAT(dim=hidden_dim, input_resolution=img_size, num_heads=nhead_2, win_size=[1, img_size[1]], token_mlp='ffn')

        self.SFT = SFT(dim=hidden_dim, input_resolution=img_size, num_heads=nhead_3, mask=[1, 8],)

        self.norm = nn.LayerNorm(hidden_dim)

        # reconstruction
        self.reconstruct = RCAN(n_feats=32)
        self.upconv1 = nn.Conv2d(hidden_dim, hidden_dim * 1, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(hidden_dim, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#        self.pixel_shuffle = nn.PixelShuffle(2)


        self.hidden_dim = hidden_dim
        self.nhead_1 = nhead_1
        self.nhead_2 = nhead_2
        self.nhead_3 = nhead_3
        self.num_attn_layers = num_attn_layers


    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, pos_enc: Optional[Tensor] = None, pos_enc_down: Optional[Tensor] = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        """

        # flatten NxCxHxW to WxHNxC
        bs, c, hn, w = feat_left.shape
        hn_down, w_down = int(hn / 2), int(w / 2)


        feat = torch.cat((feat_left, feat_right), dim=0)

        # for depthwise
        feat = self.depthwise_conv_for_self1(feat)
        feat = feat.flatten(2).permute(0, 2, 1).contiguous()
        # forward:
        for blk in self.self_attn_layers1:
            feat = blk(feat, x_size=(hn, w))
        feat = feat.permute(0, 2, 1).view(2*bs, c, hn, w).contiguous()


        feat_left = feat[:bs]
        feat_right = feat[bs:]


        # for original size
        # depthwise conv
        feat_left = self.depthwise_conv_for_cross1(feat_left)
        feat_right = self.depthwise_conv_for_cross1(feat_right)

        # positional encoding
        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None
        # forward
        feat_left = feat_left.flatten(2).permute(0, 2, 1).contiguous()
        feat_right = feat_right.flatten(2).permute(0, 2, 1).contiguous()


        # fusion
        # spatial size
        feat_left_wraped = self.SAT(query=feat_left, key=feat_right, value=feat_right, x_size=(hn, w))
        feat_right_wraped = self.SAT(query=feat_right, key=feat_left, value=feat_left, x_size=(hn, w))

        feat_left = feat_left.permute(0, 2, 1).view(bs, c, hn, w).contiguous()
        feat_right = feat_right.permute(0, 2, 1).view(bs, c, hn, w).contiguous()
        feat_left_wraped = feat_left_wraped.permute(0, 2, 1).view(bs, c, hn, w).contiguous()
        feat_right_wraped = feat_right_wraped.permute(0, 2, 1).view(bs, c, hn, w).contiguous()


        # depthwise conv
        # for left:
        feat_left = self.depthwise_conv_for_fusion(feat_left)
        feat_left_wraped = self.depthwise_conv_for_fusion(feat_left_wraped)
        # for right:
        feat_right = self.depthwise_conv_for_fusion(feat_right)
        feat_right_wraped = self.depthwise_conv_for_fusion(feat_right_wraped)
        feat_left = feat_left.flatten(2).permute(0, 2, 1).contiguous()
        feat_right = feat_right.flatten(2).permute(0, 2, 1).contiguous()
        # for wraped_feat
        feat_left_wraped = feat_left_wraped.flatten(2).permute(0, 2, 1).contiguous()
        feat_right_wraped = feat_right_wraped.flatten(2).permute(0, 2, 1).contiguous()
        # forward

        feat_left = self.SFT(query=feat_left, key=feat_left_wraped, value=feat_left_wraped, x_size=(hn, w))
        feat_right = self.SFT(query=feat_right, key=feat_right_wraped, value=feat_right_wraped, x_size=(hn, w))
        feat_left = feat_left.permute(0, 2, 1).view(bs, c, hn, w).contiguous()
        feat_right = feat_right.permute(0, 2, 1).view(bs, c, hn, w).contiguous()


        feat = torch.cat((feat_left, feat_right), dim=0)
        # depthwise conv
        feat = self.depthwise_conv_for_self2(feat)
        feat = feat.flatten(2).permute(0, 2, 1).contiguous()
        # forward:
        for blk in self.self_attn_layers2:
            feat = blk(feat, x_size=(hn, w))
        feat = feat.permute(0, 2, 1).view(2*bs, c, hn, w).contiguous()



        feat_left = feat[:bs]
        feat_right = feat[bs:]

        feat_left = self.reconstruct(feat_left)
#        feat_left = self.lrelu(self.pixel_shuffle(self.upconv1(feat_left)))
        feat_left = self.lrelu(self.upconv1(feat_left))
        out_left = self.conv_last(feat_left)

        feat_right = self.reconstruct(feat_right)
#        feat_right = self.lrelu(self.pixel_shuffle(self.upconv1(feat_right)))
        feat_right = self.lrelu(self.upconv1(feat_right))
        out_right = self.conv_last(feat_right)

        return out_left, out_right



def build_transformer(args):
    return Transformer(
        hidden_dim=args.channel_dim,
        nhead=args.nheads,
        num_attn_layers=args.num_attn_layers
    )
