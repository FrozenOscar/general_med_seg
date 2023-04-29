import torch
import torch.nn as nn
from backbone import *
from blocks import ConvBlock
from decoder import Unet_decoder


# class UpConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, is_deconv=False):
#         super(UpConvBlock, self).__init__()
#         if is_deconv:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2))
#         else:
#             self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.conv = ConvBlock(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.conv(x)
#         return x


class Unet(nn.Module):
    def __init__(self, in_channel=3, num_classes=1, backbone='unet_backbone', pretrained=False):
        super(Unet, self).__init__()
        self.channels = [64, 128, 256, 512, 1024]

        # Load backbone, no need to pass in num_classes
        backbone = create_backbone(backbone, in_channel=in_channel, out_channels=self.channels)

        self.maxpool = backbone.maxpool
        ## -------------Encoder--------------
        self.enc1 = backbone.layer0
        self.enc2 = backbone.layer1
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.center = ConvBlock(self.channels[3], self.channels[4])

        # Decoder
        self.dec4 = Unet_decoder(self.channels[3:5], cat_channels=self.channels[3], out_channels=self.channels[3])
        self.dec3 = Unet_decoder(self.channels[2:4], cat_channels=self.channels[2], out_channels=self.channels[2])
        self.dec2 = Unet_decoder(self.channels[1:3], cat_channels=self.channels[1], out_channels=self.channels[1])
        self.dec1 = Unet_decoder(self.channels[0:2], cat_channels=self.channels[0], out_channels=self.channels[0])

        # mask_head
        self.out = nn.Conv2d(self.channels[0], num_classes, kernel_size=(1, 1))

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)             # (B, 64, H//2, W//2)
        enc2 = self.enc2(self.maxpool(enc1))          # (B, 128, H//4, W//4)
        enc3 = self.enc3(self.maxpool(enc2))          # (B, 256, H//8, W//8)
        enc4 = self.enc4(self.maxpool(enc3))          # (B, 512, H//16, W//16)
        center = self.center(self.maxpool(enc4))      # (B, 1024, H//16, W//16))

        # Decoder
        dec4 = self.dec4(enc4, center)
        dec3 = self.dec3(enc3, dec4)
        dec2 = self.dec2(enc2, dec3)
        dec1 = self.dec1(enc1, dec2)

        # Output
        out = self.out(dec1)
        return out


if __name__ == '__main__':
    import time

    backbone = 'unet_backbone'
    # backbone = 'resnext50_32x4d'
    # backbone = 'resnet101'

    model = Unet(backbone=backbone, in_channel=1, num_classes=4)
    model.cuda()
    x = torch.randn((1, 1, 512, 512)).cuda()
    record = []
    _ = model(x)
    for i in range(20):
        start = time.time()
        _ = model(x)
        record.append(time.time() - start)
    print(record)
    print(f'Average inference time: {sum(record)/len(record):.4f}')

    # x = torch.zeros((1, 1, 512, 512))
    # y = model(x)
    # print(y.shape)

    # # 加载权重文件
    # model_state_dict = torch.load('../checkpoints/Unet_unet-backbone_epoch20_batch8_cosine_2e-4.pth',
    #                               map_location='cpu')
    #
    # # 构建模型并加载权重·
    # model = Unet(backbone='unet_backbone', in_channel=1, num_classes=5)
    # model.load_state_dict(model_state_dict['net'])
    # a = 0

    # from torchsummary import summary
    #
    # print("模型参数量：", sum(p.numel() for p in model.parameters()))  # 打印模型参数量
    # summary(model, input_size=(1, 512, 512))  # 使用 torchsummary 打印模型概况，包括参数数量
