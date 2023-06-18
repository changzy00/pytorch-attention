""" 
PyTorch implementation of EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

As described in http://arxiv.org/abs/1905.11946

The authors propose a new scaling method that uniformly scales all dimensions of
depth/width/resolution using a simple yet highly effective compound coefficient.
"""

import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
                            )
    
    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avgpool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)
    
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=in_channels * expand_ratio)
        self.bn2 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.se = SELayer(in_channels * expand_ratio)
        self.conv3 = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += x
        return out
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        in_planes = 32
        last_planes = 1280
        inverted_residual_setting = [
                # t, c, n, s, k
                [1, 16, 1, 1, 3],
                [6, 24, 2, 2, 3],
                [6, 40, 2, 2, 5],
                [6, 80, 3, 2, 3],
                [6, 112, 3, 1, 5],
                [6, 192, 4, 2, 5],
                [6, 320, 1, 1, 3],
            ]
        features = [nn.Sequential(
            nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU6(inplace=True)
        )]
        for t, c, n, s, k in inverted_residual_setting:
            out_planes = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedBottleneck(in_planes, out_planes, k, stride, t))
                in_planes = out_planes
        features.append(nn.Sequential(
            nn.Conv2d(in_planes, last_planes, kernel_size=1),
            nn.BatchNorm2d(last_planes),
            nn.ReLU6(inplace=True)
        ))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(last_planes, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.head(out)
        return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = EfficientNet()
    y = model(x)
    print(y.shape)




