""" 
PyTorch implementation of MobileNetV2: Inverted Residuals and Linear Bottlenecks

As described in https://arxiv.org/abs/1801.04381

The MobileNetV2 architecture is based on an inverted residual structure where the input 
and output of the residual block are thin bottleneck layers opposite to traditional residual 
models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise 
convolutions to filter features in the intermediate expansion layer.
"""





import torch
from torch import nn

class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3,
                               stride=stride, padding=1, groups=in_channels * expand_ratio)
        self.bn2 = nn.BatchNorm2d(in_channels * expand_ratio)

        self.conv3 = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += x
        return out
    
class MobileNetv2(nn.Module):
    def __init__(self, drop=0.2, num_classes=1000):
        super().__init__()
        in_planes = 32
        last_planes = 1280
        inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        features = [nn.Sequential(
            nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU6(inplace=True)
        )]
        for t, c, n, s in inverted_residual_setting:
            out_planes = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedBottleneck(in_planes, out_planes, stride, t))
                in_planes = out_planes
        features.append(nn.Sequential(
            nn.Conv2d(in_planes, last_planes, kernel_size=1),
            nn.BatchNorm2d(last_planes),
            nn.ReLU6(inplace=True)
        ))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(drop),
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
    model = MobileNetv2()
    y = model(x)
    print(y.shape)