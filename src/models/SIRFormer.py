#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer
from .rcan import *


class SIRFormer(nn.Module):

    def __init__(self, args):
        super(SIRFormer, self).__init__()

        # self.feat_extraction = PatchEmbed(
        #     img_size=(256, 512), patch_size=args.patch_size, in_chans=3, embed_dim=args.channel_dim)
        self.head = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.feature_extraction = RCAN(n_feats=32)
        self.transformer = Transformer(img_size = args.patch_size, hidden_dim = args.channel_dim)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, left_img, right_img):
        """
        :param x: input data
        :return:
            a dictionary object with keys
            - "disp_pred" [N,H,W]: predicted disparity
            - "occ_pred" [N,H,W]: predicted occlusion mask
            - "disp_pred_low_res" [N,H//s,W//s]: predicted low res (raw) disparity
        """

        feat_left = self.head(left_img)
        feat_right = self.head(right_img)
        feat_left = self.feature_extraction(feat_left)
        feat_right = self.feature_extraction(feat_right)
        left_up = F.interpolate(left_img, scale_factor=2, mode='bicubic', align_corners=False)
        right_up = F.interpolate(right_img, scale_factor=2, mode='bicubic', align_corners=False)
        
        # pos_enc = self.pos_encoder(feat_left)  # 2W-1*C
        # pos_enc_down = self.pos_encoder(self.downsample(feat_left))
        out_left_sr, out_right_sr = self.transformer(feat_left, feat_right, None, None)
        left_sr = out_left_sr + left_up
        right_sr = out_right_sr + right_up

        left_sr = torch.clamp(left_sr, 0., 1.)
        right_sr = torch.clamp(right_sr, 0., 1.)

        return left_sr, right_sr
