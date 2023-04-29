import torch
import torch.nn as nn


# 该处与AttUnet论文上的注意力单元完全一样
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l):
        super(Attention_block, self).__init__()
        F_int = F_g // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


# SE模块（通道注意力）
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)      # 因为avg_pool输出维度是(b, c, 1, 1), 所以可以这么写
        y = self.fc(y).reshape(b, c, 1, 1)      # 相当于用了一个MLP给每个通道赋予了一个权重
        return x * y.expand_as(x)


class Attention_block_SE(nn.Module):
    def __init__(self, F_g, F_l):
        super(Attention_block_SE, self).__init__()
        F_int = F_g

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            SELayer(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            SELayer(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 512, 64, 64)
    att = Attention_block_SE(F_g=512, F_l=512)
    # att = Attention_block(F_g=512, F_l=512)
    y = att(x, x)
    print(y.shape)

