import torch
import torch.nn as nn
from backbone import *
from blocks import *
from decoder import AttUnet_decoder


# class UpAttConv(nn.Module):
#     def __init__(self, in_channels, out_channels, att_block):
#         # in_channels是上采样前的channel[1024, 512, 256, 128], out_channels则对应的是[512, 256, 128, 64]
#         super(UpAttConv, self).__init__()
#         self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
#                                 nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
#                                 nn.BatchNorm2d(out_channels),
#                                 nn.ReLU(inplace=True),)
#         self.att = att_block(out_channels, out_channels)
#         self.conv = ConvBlock(out_channels*2, out_channels)
#
#     def forward(self, dec, enc):
#         # center或dec先上采样得x1，x1和enc进入att, 得到x2, 接着cat([x1, x2])接两个conv即可
#         x1 = self.up(dec)
#         x2 = self.att(x1, enc)
#         y = self.conv(torch.cat((x1, x2), dim=1))
#         return y


class AttUnet(nn.Module):
    def __init__(self, in_channel=3, backbone='resnet18', attn_block=None, pretrained=False, num_classes=1):
        super(AttUnet, self).__init__()
        self.channels = (64, 128, 256, 512, 1024)
        self.attn_block = attn_block
        if not self.attn_block:
            self.attn_block = Attention_block

        # Load backbone, no need to pass in num_classes
        backbone = create_backbone(backbone, in_channel=in_channel, out_channels=self.channels)

        self.maxpool = backbone.maxpool
        # Encoder
        self.enc1 = backbone.layer0
        self.enc2 = backbone.layer1
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.center = ConvBlock(self.channels[3], self.channels[4])

        # Decoder
        self.dec4 = AttUnet_decoder(self.channels[4], self.channels[3], self.channels[3], self.attn_block)
        self.dec3 = AttUnet_decoder(self.channels[3], self.channels[2], self.channels[2], self.attn_block)
        self.dec2 = AttUnet_decoder(self.channels[2], self.channels[1], self.channels[1], self.attn_block)
        self.dec1 = AttUnet_decoder(self.channels[1], self.channels[0], self.channels[0], self.attn_block)

        # mask_head
        self.out = nn.Conv2d(self.channels[0], num_classes, kernel_size=(1, 1))

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)             # (B, 64, H//2, W//2)
        enc2 = self.enc2(self.maxpool(enc1))  # (B, 128, H//4, W//4)
        enc3 = self.enc3(self.maxpool(enc2))  # (B, 256, H//8, W//8)
        enc4 = self.enc4(self.maxpool(enc3))  # (B, 512, H//16, W//16)
        center = self.center(self.maxpool(enc4))      # (B, 1024, H//16, W//16))

        # Decoder
        dec4 = self.dec4(center, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        # Output
        out = self.out(dec1)
        return out


if __name__ == '__main__':
    backbone = 'unet_backbone'
    # backbone = 'resnext50_32x4d'
    # backbone = 'resnet101'
    # backbone = 'resnet34'

    model = AttUnet(backbone=backbone, attn_block=Attention_block, in_channel=1, num_classes=4)
    x = torch.zeros((1, 1, 512, 512))
    y = model(x)
    print(y.shape)

    # model_state_dict = torch.load('../checkpoints/AttUnet_unet-backbone_epoch20_batch8_cosine_2e-4.pth',
    #                               map_location='cpu')
    #
    # # 构建模型并加载权重
    # model = AttUnet(backbone='unet_backbone', in_channel=1, num_classes=5)
    # model.load_state_dict(model_state_dict['net'])
    # a = 0
