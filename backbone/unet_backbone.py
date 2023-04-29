import torch
import torch.nn as nn
from inits import init_weights
from blocks.conv import ConvBlock


class UnetBackbone(nn.Module):
    # 最原始的Unet backbone
    def __init__(self, in_channel=3):
        super(UnetBackbone, self).__init__()
        channels = [64, 128, 256, 512]

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.layer0 = ConvBlock(in_channel, channels[0])
        self.layer1 = ConvBlock(channels[0], channels[1])
        self.layer2 = ConvBlock(channels[1], channels[2])
        self.layer3 = ConvBlock(channels[2], channels[3])

    def forward(self, x, out_all=False):
        enc1 = self.layer0(x)
        enc2 = self.layer1(self.maxpool(enc1))
        enc3 = self.layer2(self.maxpool(enc2))
        enc4 = self.layer3(self.maxpool(enc3))

        if out_all:
            return [enc1, enc2, enc3, enc4]
        else:
            return enc4


if __name__ == '__main__':
    model = UnetBackbone(in_channel=1)
    # x = torch.zeros((1, 3, 512, 512))
    # y = model(x)
    # print(y.shape)

    model = model.cuda()
    from torchsummary import summary

    print("模型参数量：", sum(p.numel() for p in model.parameters()))    # 打印模型参数量
    summary(model, input_size=(1, 256, 256))        # 使用 torchsummary 打印模型概况，包括参数数量
