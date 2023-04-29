import torch
import torch.nn as nn
from collections import OrderedDict
from inits import *


# class ConvBlock(nn.Module):
#     """
#         连续两个3x3卷积, 不改变尺寸, in_chan->out_chan, out_chan->out_chan
#     """
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.two_conv = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)),
#             ('bn1', nn.BatchNorm2d(out_channels)),
#             ('act1', nn.ReLU()),
#             ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)),
#             ('bn2', nn.BatchNorm2d(out_channels)),
#             ('act2', nn.ReLU()),
#         ]))
#
#     def forward(self, x):
#         x = self.two_conv(x)
#         return x

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    s = stride
    p = dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(s, s),
        padding=p, groups=groups, bias=False, dilation=(p, p),)


# class ConvBlock(nn.Module):
#     """
#         conv + batch_norm + relu
#     """
#     def __init__(self, in_channels, out_channels, is_batchnorm=True, n=2, ks=3, stride=1, padding=1):
#         super(ConvBlock, self).__init__()
#         self.n = n          # n=2时进行两次卷积
#         self.ks = ks
#         self.stride = stride
#         self.padding = padding
#         s = stride
#         p = padding
#         if is_batchnorm:
#             for i in range(1, n + 1):
#                 conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (ks, ks), (s, s), p),
#                                      nn.BatchNorm2d(out_channels),
#                                      nn.ReLU(), )
#                 setattr(self, 'conv%d' % i, conv)       # 为conv1等属性设置属性值
#                 in_channels = out_channels
#
#         else:
#             for i in range(1, n + 1):
#                 conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (ks, ks), (s, s), p),
#                                      nn.ReLU(), )
#                 setattr(self, 'conv%d' % i, conv)
#                 in_channels = out_channels
#
#         # initialise the blocks
#         for m in self.children():
#             init_weights(m, init_type='kaiming')
#
#     def forward(self, x):
#         for i in range(1, self.n + 1):
#             conv = getattr(self, 'conv%d' % i)
#             x = conv(x)
#         return x


def ConvBlock(in_channels, out_channels, is_batchnorm=True, n=2, ks=3, stride=1, padding=1):
    """
        n * (conv + batch_norm + relu)
    """
    s = stride
    p = padding
    conv_block = []
    if is_batchnorm:
        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (ks, ks), (s, s), p),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(), )
            in_channels = out_channels
            conv_block.append(conv)
    else:
        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (ks, ks), (s, s), p),
                                 nn.ReLU(), )
            in_channels = out_channels
            conv_block.append(conv)
    return nn.Sequential(*conv_block)


if __name__ == '__main__':
    pass
