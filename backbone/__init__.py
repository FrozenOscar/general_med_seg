from .VGG import VGGBackbone
from .resnet import ResnetBackbone, ResneXtBackbone
from .unet_backbone import UnetBackbone


def create_backbone(backbone, in_channel, **kwargs):
    if backbone == 'unet_backbone':
        backbone = UnetBackbone(in_channel=in_channel)
    elif 'vgg' in backbone:
        backbone = VGGBackbone(backbone, in_channel=in_channel)
    elif 'resnet' in backbone:
        backbone = ResnetBackbone(backbone, in_channel=in_channel, **kwargs)
    elif 'resnext' in backbone:
        backbone = ResneXtBackbone(backbone, in_channel=in_channel, **kwargs)
    return backbone
