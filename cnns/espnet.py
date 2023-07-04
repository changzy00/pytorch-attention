""" 
PyTorch implementation of ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

As described in https://arxiv.org/abs/1803.06815

ESP is based on a convolution factorization principle that decomposes a standard
convolution into two steps: (1) point-wise convolutions and (2) spatial pyramid of dilated convolutions.
"""





import torch
from torch import nn

class ESPBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        n = planes // 4
        self.downsample = inplanes != planes

        if self.downsample:
            self.conv1 = nn.Conv2d(inplanes, n, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(inplanes, n, 1)
        self.k1 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=3//2*1, dilation=1)
        self.k2 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=3//2*2, dilation=2)
        self.k3 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=3//2*4, dilation=4)
        self.k4 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=3//2*8, dilation=8)
        self.bn = nn.BatchNorm2d(planes)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        k1 = self.k1(out)
        k2 = self.k2(out)
        k3 = self.k3(out)
        k4 = self.k4(out)

        a2 = k1 + k2
        a3 = a2 + k3
        a4 = a3 + k4
        out = torch.cat([k1, a2, a3, a4], dim=1)
        out = self.bn(out)
        out = self.act(out)
        if not self.downsample:
            out += x
        return out
    
class ESPNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.espblock_1 = ESPBlock(16, 64)
        self.espblock_2 = nn.Sequential(*[ESPBlock(64, 64) for i in range(2)])
        self.espblock_3 = ESPBlock(64, 128)
        self.espblock_4 = nn.Sequential(*[ESPBlock(128, 128) for i in range(3)])
        self.head = nn.Conv2d(128, num_classes, 1)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.espblock_1(x)
        x = self.espblock_2(x)
        x = self.espblock_3(x)
        x = self.espblock_4(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = ESPNet(10)
    y = model(x)
    print(y.shape)
