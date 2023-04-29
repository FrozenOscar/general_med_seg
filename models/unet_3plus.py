import torch
import torch.nn as nn
from backbone import create_backbone
from blocks import ConvBlock
from decoder import Unet_decoder


class Unet3plus(nn.Module):

    def __init__(self, in_channel=3, num_classes=1, backbone='unet_backbone', upsample_mode='bilinear', is_batchnorm=True):
        super(Unet3plus, self).__init__()
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


        ## -------------Decoder--------------
        cat_channels = self.channels[0]
        num_cat = 5
        cat_out_channels = cat_channels * num_cat
        self.dec4 = Unet_decoder([*self.channels[:4], self.channels[-1]], cat_channels=cat_channels,
                                      out_channels=cat_out_channels, enc_num=3, center_num=1, dec_num=1,
                                      upsample_mode=upsample_mode, is_batchnorm=is_batchnorm)
        self.dec3 = Unet_decoder([*self.channels[:3], cat_out_channels, self.channels[-1]], cat_channels=cat_channels,
                                      out_channels=cat_out_channels, enc_num=2, center_num=1, dec_num=2,
                                      upsample_mode=upsample_mode, is_batchnorm=is_batchnorm)
        self.dec2 = Unet_decoder([*self.channels[:2], *[cat_out_channels]*2, self.channels[-1]], cat_channels=cat_channels,
                                      out_channels=cat_out_channels, enc_num=1, center_num=1, dec_num=3,
                                      upsample_mode=upsample_mode, is_batchnorm=is_batchnorm)
        self.dec1 = Unet_decoder([*self.channels[:1], *[cat_out_channels]*3, self.channels[-1]], cat_channels=cat_channels,
                                      out_channels=cat_out_channels, enc_num=0, center_num=1, dec_num=4,
                                      upsample_mode=upsample_mode, is_batchnorm=is_batchnorm)
        self.out = nn.Conv2d(num_cat * cat_channels, num_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # -------------Encoder-------------
        enc1 = self.enc1(x)             # (B, 64, H//2, W//2)
        enc2 = self.enc2(self.maxpool(enc1))          # (B, 128, H//4, W//4)
        enc3 = self.enc3(self.maxpool(enc2))          # (B, 256, H//8, W//8)
        enc4 = self.enc4(self.maxpool(enc3))          # (B, 512, H//16, W//16)
        center = self.center(self.maxpool(enc4))      # (B, 1024, H//16, W//16))

        # # -------------Decoder-------------
        dec4 = self.dec4(enc1, enc2, enc3, enc4, center)
        dec3 = self.dec3(enc1, enc2, enc3, dec4, center)
        dec2 = self.dec2(enc1, enc2, dec3, dec4, center)
        dec1 = self.dec1(enc1, dec2, dec3, dec4, center)

        out = self.out(dec1)  # d1->320*320*n_classes
        return torch.sigmoid(out)


if __name__ == '__main__':
    backbone = 'unet_backbone'
    model = UNet_3Plus(backbone='unet_backbone', in_channel=1, num_classes=4)
    model = model.cuda()
    # x = torch.zeros((1, 1, 256, 256))
    # # x = x.cuda()
    # y = model(x)
    # print(y.shape)

    from torchsummary import summary

    print("模型参数量：", sum(p.numel() for p in model.parameters()))    # 打印模型参数量
    summary(model, input_size=(1, 256, 256))        # 使用 torchsummary 打印模型概况，包括参数数量
