from blocks import *


class Unet3plus_decoder(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels,
                 enc_num=0, center_num=1, dec_num=1, **kwargs):
        """
        Args:
            in_channels (List[int]): 所有要concat的层特征图的通道数
            cat_channels (int): 每层concat时要转换成的通道数（例如Unet 3+, 每层concat前通道数都转为64）
            out_channels (int): 当前decoder层输出特征图通道数
            **kwargs:
        """
        super(Unet3plus_decoder, self).__init__()
        self.in_channels = [in_channels] if not isinstance(in_channels, list) else in_channels
        self.enc_num = enc_num
        self.center_num = center_num
        self.dec_num = dec_num
        self.cat_num = enc_num + center_num + dec_num

        upsample_mode = kwargs['upsample_mode'] if 'upsample_mode' in kwargs.keys() else 'bilinear'
        is_batchnorm = kwargs['is_batchnorm'] if 'is_batchnorm' in kwargs.keys() else True

        maxpool_scale = [2 ** i for i in range(1, enc_num+1)][::-1]
        upsample_scale = [2 ** i for i in range(1, dec_num+1)]

        idx = 0
        # ------------- Encoder部分concat前操作 -------------
        enc_conv_list = []
        for i in range(enc_num):
            conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=maxpool_scale[i], ceil_mode=True),
                ConvBlock(self.in_channels[idx], cat_channels, n=1, is_batchnorm=is_batchnorm)
            )
            # enc_conv_list.append(*max_pool)
            enc_conv_list.append(conv)
            idx += 1
        self.enc_conv_list = nn.Sequential(*enc_conv_list)

        # ------------- center部分（同一层）concat前操作 -------------
        self.center_conv = ConvBlock(self.in_channels[idx], cat_channels, n=1, is_batchnorm=is_batchnorm)
        idx += 1

        # ------------- Decoder部分concat前操作 -------------
        dec_conv_list = []
        for i in range(dec_num):
            conv = nn.Sequential(
                UpSample(self.in_channels[idx], cat_channels, upsample_mode=upsample_mode, scale=upsample_scale[i]),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU()
            )
            dec_conv_list.append(conv)
            idx += 1
        self.dec_conv_list = nn.Sequential(*dec_conv_list)

        # ---------- concat以后的conv -------------
        self.cat_conv = ConvBlock(cat_channels*self.cat_num, out_channels, n=1, is_batchnorm=is_batchnorm)

    def forward(self, *x_list):
        assert len(x_list) == self.cat_num
        outputs = []
        idx = 0

        # ------------- Encoder -------------
        for i in range(self.enc_num):
            outputs.append(self.enc_conv_list[i](x_list[idx]))
            idx += 1

        # ------------- center -------------
        outputs.append(self.center_conv(x_list[idx]))
        idx += 1

        # ------------- decoder -------------
        for i in range(self.dec_num):
            outputs.append(self.dec_conv_list[i](x_list[idx]))
            idx += 1

        res = self.cat_conv(torch.cat(outputs, dim=1))
        return res


if __name__ == '__main__':
    # channels = [64, 128, 320, 320, 1024]
    channels = [64, 128, 256, 512, 1024]

    cat_channels = channels[0]
    num_cat = 5
    cat_out_channels = cat_channels * num_cat
    model = Unet3plus_decoder([*channels], cat_channels=cat_channels,
                             out_channels=cat_out_channels, enc_num=3, center_num=1, dec_num=1)
    model = model.cuda()

    from torchsummary import summary

    print("模型参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))    # 打印模型参数量
    # input_sizes = [(64, 256, 256), (128, 128, 128), (320, 64, 64), (320, 32, 32), (1024, 16, 16)]
    input_sizes = [(64, 256, 256), (128, 128, 128), (256, 64, 64), (512, 32, 32), (1024, 16, 16)]
    summary(model, input_size=input_sizes)        # 使用 torchsummary 打印模型概况，包括参数数量