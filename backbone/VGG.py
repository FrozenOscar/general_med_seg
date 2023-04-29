import torch
import torch.nn as nn
from collections import OrderedDict


def make_layers(cfg, in_channel=3, batch_norm=False) -> nn.Sequential:
    backbone = []
    layer = []
    for v in cfg:
        if v == "M":
            backbone.append(layer)
            layer = []
            # layer = [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channel, int(v), kernel_size=(3, 3), padding=1)
            if batch_norm:
                layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]
            in_channel = v
    backbone = nn.Sequential(OrderedDict([
        ('layer0', nn.Sequential(*backbone[0])), ('layer1', nn.Sequential(*backbone[1])),
        ('layer2', nn.Sequential(*backbone[2])), ('layer3', nn.Sequential(*backbone[3])),
        ('layer4', nn.Sequential(*backbone[4])),]))
    return backbone


class VGG(nn.Module):
    """
        每个stage自身都不带maxpool操作, maxpool操作在forward函数中添加
    """
    def __init__(self, all_layers, init_weights: bool = True, dropout: float = 0.5):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer0 = all_layers.layer0
        self.layer1 = all_layers.layer1
        self.layer2 = all_layers.layer2
        self.layer3 = all_layers.layer3
        self.layer4 = all_layers.layer4

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, out_all=False):
        enc1 = self.layer0(x)
        enc2 = self.layer1(self.maxpool(enc1))
        enc3 = self.layer2(self.maxpool(enc2))
        enc4 = self.layer3(self.maxpool(enc3))
        enc5 = self.layer4(self.maxpool(enc4))

        if out_all:
            return [enc1, enc2, enc3, enc4, enc5]
        else:
            return enc5


def VGGBackbone(backbone, **kwargs) -> VGG:
    cfgs = {
        "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }

    if 'batch_norm' in kwargs.keys():
        batch_norm = kwargs['batch_norm'] if kwargs['batch_norm'] is not None else True
        kwargs.pop('batch_norm')
    else:
        batch_norm = True

    if 'in_channel' in kwargs.keys():
        in_channel = kwargs['in_channel'] if kwargs['in_channel'] is not None else 3
        kwargs.pop('in_channel')
    else:
        in_channel = 3

    if backbone == 'vgg11':
        return VGG(make_layers(cfgs['A'], in_channel=in_channel, batch_norm=batch_norm), **kwargs)
    elif backbone == 'vgg13':
        return VGG(make_layers(cfgs['B'], in_channel=in_channel, batch_norm=batch_norm), **kwargs)
    elif backbone == 'vgg16':
        return VGG(make_layers(cfgs['D'], in_channel=in_channel, batch_norm=batch_norm), **kwargs)
    elif backbone == 'vgg19':
        return VGG(make_layers(cfgs['E'], in_channel=in_channel, batch_norm=batch_norm), **kwargs)
    else:
        raise KeyError(f'{backbone} is not in ["vgg11", "vgg13", "vgg16", "vgg19"]')


if __name__ == '__main__':
    vgg_backbone = VGGBackbone('vgg16', in_channel=1)
    x = torch.randn(1, 1, 512, 512)
    y = vgg_backbone(x)
    a = 0
