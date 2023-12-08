import torch.nn as nn
import torch

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, input_nfeat, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.1, inplace=True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(input_nfeat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        # res += x                                  # use dense to replace residual
        return torch.cat((x, res), dim=1)

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, input_nfeat, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, input_nfeat+i*n_feat, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.1, inplace=True), res_scale=1) \
            for i in range(n_resblocks)]
        modules_body.append(conv(input_nfeat+n_resblocks*n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# default conv
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, n_resgroups=3, n_resblocks=4, reduction=2, input_n_feat=32, n_feats=32, conv=default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction

        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # define body module
        modules_body = [
            ResidualGroup(
                conv, input_n_feat, n_feats, kernel_size, reduction, act=act, res_scale=None, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        return out

