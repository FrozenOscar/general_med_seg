from blocks import *


class AttUnet_decoder(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, att_block, **kwargs):
        # in_channels是上采样前的channel[1024, 512, 256, 128], out_channels则对应的是[512, 256, 128, 64]
        super(AttUnet_decoder, self).__init__()
        self.up = UpSample(in_channels, cat_channels, **kwargs)
        self.att = att_block(cat_channels, cat_channels)
        self.cat_conv = CatConv(cat_channels, out_channels, cat_num=2, **kwargs)

    def forward(self, dec, enc):
        # center或dec先上采样得x1，x1和enc进入att, 得到x2, 接着cat([x1, x2])接两个conv即可
        x1 = self.up(dec)
        x2 = self.att(x1, enc)
        y = self.cat_conv(torch.cat((x1, x2), dim=1))
        return y


if __name__ == '__main__':
    channels = [64, 128, 256, 512, 1024]
