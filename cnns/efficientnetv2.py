""" 
PyTorch implementation of EfficientNetV2: Smaller Models and Faster Training

As described in https://arxiv.org/abs/2104.00298

This paper introduces EfficientNetV2, a new family of convolutional networks that have faster
training speed and better parameter efficiency than previous models. The models were searched 
from the search space enriched with new ops such as Fused-MBConv.
"""

import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channels, ratio=4):
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
    
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3,
                               stride=stride, padding=1, groups=in_channels * expand_ratio)
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
    
class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.relu = nn.ReLU6(inplace=True)
        self.se = SELayer(in_channels * expand_ratio)
        self.conv2 = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(out)
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            out += x
        return out
    
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        in_planes = 24
        last_planes = 1280
        inverted_residual_setting = [
                # t, c, n, s, use_fmb
                [1, 24, 2, 1, "FMB"],
                [4, 48, 4, 2, "FMB"],
                [4, 64, 4, 2, "FMB"],
                [4, 128, 6, 2, "MB"],
                [6, 160, 9, 1, "MB"],
                [6, 256, 15, 2, "MB"],
            ]
        features = [nn.Sequential(
            nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU6(inplace=True)
        )]
        for t, c, n, s, type in inverted_residual_setting:
            out_planes = c
            for i in range(n):
                stride = s if i == 0 else 1
                if type == "FMB":
                    features.append(FusedMBConv(in_planes, out_planes, stride, t))
                else:
                    features.append(MBConv(in_planes, out_planes, stride, t))
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
    model = EfficientNetV2()
    y = model(x)
    print(y.shape)




