""" 
PyTorch implementation of Rethinking Bottleneck Structure for Efficient Mobile Network Design

As described in https://arxiv.org/pdf/2007.02269.pdf

Sandglass block, that performs identity mapping and spatial transformation at higher dimensions
and thus alleviates information loss and gradient confusion effectively.
"""



import torch
from torch import nn

class SandGlassBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, reduce_ratio=6):
        super().__init__()
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, groups=inplanes),
            nn.BatchNorm2d(inplanes),
            nn.ReLU6()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduce_ratio, 1),
            nn.BatchNorm2d(inplanes // reduce_ratio)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes // reduce_ratio, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU6()
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride, 1, groups=planes),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = stride == 1 and inplanes == planes

    def forward(self, x):
        out = self.dwconv1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dwconv2(out)
        if self.shortcut:
            out += x
        return out

class MobileNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        inplanes = 32
        last_planes = 1280
        # t, c, s, n
        block_setting = [
            [2, 96, 2, 1],
            [6, 114, 1, 1],
            [6, 192, 2, 3],
            [6, 288, 2, 3],
            [6, 384, 1, 4],
            [6, 576, 2, 4],
            [6, 960, 1, 2],
            [6, 1280, 1, 1]
        ]
        features = [nn.Sequential(
            nn.Conv2d(3, inplanes, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )]
        for t, c, s, n in block_setting:
            out_planes = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(SandGlassBlock(inplanes, out_planes, stride, t))
                inplanes = out_planes
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.conv = nn.Conv2d(last_planes, num_classes, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.flatten(1)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = MobileNeXt()
    y = model(x)
    print(y.shape)
