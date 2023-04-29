import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation,)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 2      # 为了不让通道数减少太多, 所以由4设为了2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
        每个stage自身都不带maxpool操作, maxpool操作在forward函数中添加
    """
    def __init__(
        self,
        block,
        layers,
        in_channel=3,
        num_classes = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        out_channels = (64, 128, 256, 512, 1024),   # 每个stage输出的通道数
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.layer1 = self._make_layer(block, out_channels[1] // block.expansion, layers[0])        # 输出通道为64*block.expansion, 这里设为2
        self.layer2 = self._make_layer(block, out_channels[2] // block.expansion, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, out_channels[3] // block.expansion, layers[2], stride=1, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, out_channels[4] // block.expansion, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x, out_all=False):
        # x = self.conv1(x)
        # x = self.bn1(self.conv1(x))
        # enc1 = self.relu(self.bn1(self.conv1(x)))
        enc1 = self.layer0(x)
        enc2 = self.layer1(self.maxpool(enc1))
        enc3 = self.layer2(self.maxpool(enc2))
        enc4 = self.layer3(self.maxpool(enc3))
        enc5 = self.layer4(self.maxpool(enc4))

        if out_all:
            return [enc1, enc2, enc3, enc4, enc5]
        else:
            return enc5

    def forward(self, x, out_all=False):
        return self._forward_impl(x, out_all=out_all)


def ResnetBackbone(backbone, **kwargs) -> ResNet:
    if backbone == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif backbone == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif backbone == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif backbone == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif backbone == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    else:
        raise KeyError(f'{backbone} is not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]')


def _ovewrite_named_param(kwargs, param: str, new_value) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def ResneXtBackbone(backbone, **kwargs) -> ResNet:
    if backbone == 'resnext50_32x4d':
        _ovewrite_named_param(kwargs, "groups", 32)
        _ovewrite_named_param(kwargs, "width_per_group", 4)
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif backbone == 'resnext101_32x8d':
        _ovewrite_named_param(kwargs, "groups", 32)
        _ovewrite_named_param(kwargs, "width_per_group", 8)
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif backbone == 'resnext101_64x4d':
        _ovewrite_named_param(kwargs, "groups", 64)
        _ovewrite_named_param(kwargs, "width_per_group", 4)
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    else:
        raise KeyError(f'{backbone} is not in ["resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"],'
                       f' which available in torchvision.models')


if __name__ == '__main__':
    # models = ResnetBackbone('resnet101', num_classes=4)
    # models = ResneXtBackbone('resnext50_32x4d', num_classes=4)
    models = ResnetBackbone('resnet50', num_classes=4)


    x = torch.zeros((1, 3, 512, 512))
    y = models(x)
    print(y.shape)
