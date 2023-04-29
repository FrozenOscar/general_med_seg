import torch
import torch.nn as nn
from blocks.conv import *
from collections import OrderedDict


# class UpSample(nn.Module):
#     def __init__(self, in_channels, out_channels, upsample_mode='bilinear', scale=2):
#         super(UpSample, self).__init__()
#         if upsample_mode == 'conv':
#             self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(scale, scale), stride=(scale, scale))
#         else:
#             self.up = nn.Sequential(
#                 nn.Upsample(scale_factor=scale, mode=upsample_mode),
#                 nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
#
#     def forward(self, x):
#         return self.up(x)


def UpSample(in_channels, out_channels, upsample_mode='bilinear', scale=2):
    if upsample_mode == 'conv':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(scale, scale), stride=(scale, scale))
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=upsample_mode),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))


class CatConv(nn.Module):
    def __init__(self, cat_channels, out_channels, cat_num=2, conv_num=1, ks=3, padding=1):
        super(CatConv, self).__init__()
        self.conv = ConvBlock(cat_channels*cat_num, out_channels, n=conv_num, ks=ks, padding=padding)

    def forward(self, *x):
        return self.conv(torch.cat([*x], 1))


if __name__ == '__main__':
    pass

