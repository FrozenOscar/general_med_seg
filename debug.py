import timm
import torch.nn as nn
import nibabel as nib
import numpy as np


# 创建resnet101模型
# model = timm.create_model('resnet101', pretrained=False)
# a = 0
# 获取模型配置信息
# config = model.default_cfg

# 提取每个stage
# stage0 = nn.Sequential(*list(model.children())[:config['stage0']['num_blocks']])
# stage1 = nn.Sequential(*list(model.children())[config['stage0']['num_blocks']:config['stage1']['num_blocks']])
# stage2 = nn.Sequential(*list(model.children())[config['stage1']['num_blocks']:config['stage2']['num_blocks']])
# stage3 = nn.Sequential(*list(model.children())[config['stage2']['num_blocks']:config['stage3']['num_blocks']])
# stage4 = nn.Sequential(*list(model.children())[config['stage3']['num_blocks']:])

# nii_data = nib.load('../data/MICCAI_pre_test_data/Subtask1/TrainMask/train_0002.nii.gz')
# img_vol = nii_data.get_fdata()
# a = 0

class ConvBlock(nn.Module):
    """
        conv + batch_norm + relu
    """
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        s, p = stride, padding
        self.conv = nn.Conv2d(in_channels, out_channels, (ks, ks), (s, s), p)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def conv_efficiency_test():
    from torchsummary import summary

    model1 = nn.Sequential(
        # nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    model1 = model1.cuda()

    print("模型参数量：", sum(p.numel() for p in model1.parameters()))  # 打印模型参数量
    summary(model1, input_size=(64, 256, 256))  # 使用 torchsummary 打印模型概况，包括参数数量

    model2 = nn.Sequential(
        # nn.MaxPool2d(2, 2, ceil_mode=True),
        ConvBlock(64, 64),
    )
    model2 = model2.cuda()

    print("模型参数量：", sum(p.numel() for p in model2.parameters()))  # 打印模型参数量
    summary(model2, input_size=(64, 256, 256))  # 使用 torchsummary 打印模型概况，包括参数数量


    model3 = ConvBlock(64, 64)
    model3 = model3.cuda()

    print("模型参数量：", sum(p.numel() for p in model3.parameters()))  # 打印模型参数量
    summary(model3, input_size=(64, 256, 256))  # 使用 torchsummary 打印模型概况，包括参数数量


def visualize_cal_graph():
    import torchviz
    import torch

    model = ConvBlock(64, 64)
    x = torch.randn(1, 64, 256, 256, requires_grad=True)
    y = model(x)
    dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
    dot.render('conv_block', format='png')

    model2 = nn.Sequential(
        ConvBlock(64, 64),
    )
    x = torch.randn(1, 64, 256, 256, requires_grad=True)
    y = model2(x)
    dot = torchviz.make_dot(y, params=dict(model2.named_parameters()))
    dot.render('conv_block_seq', format='png')



if __name__ == '__main__':
    # conv_efficiency_test()
    visualize_cal_graph()
