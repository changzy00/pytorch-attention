""" 
PyTorch implementation of Wide Residual Networks

As described in https://arxiv.org/pdf/1605.07146

The authors propose a novel widened architecture for ResNet blocks that allows for residual
networks with significantly improved performance.
"""



import torch
from torch import nn


class WideBasic(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):

        residual = self.residual(x)
        shortcut = self.shortcut(x)

        return residual + shortcut

class WideResNet(nn.Module):
    def __init__(self, num_classes, block, depth=50, widen_factor=1):
        super().__init__()

        self.depth = depth
        k = widen_factor
        l = int((depth - 4) / 6)
        self.in_channels = 16
        self.init_conv = nn.Conv2d(3, self.in_channels, 3, 1, padding=1)
        self.conv2 = self._make_layer(block, 16 * k, l, 1)
        self.conv3 = self._make_layer(block, 32 * k, l, 2)
        self.conv4 = self._make_layer(block, 64 * k, l, 2)
        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * k, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)


def wideresnet(depth=40, widen_factor=10):
    net = WideResNet(100, WideBasic, depth=depth, widen_factor=widen_factor)
    return net

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = wideresnet()
    y = model(x)
    print(y.shape)
