""" 
PyTorch implementation of MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications

As described in https://arxiv.org/pdf/1704.04861

MobileNets are based on a streamlined architecture that uses depthwise separable convolutions 
to build light weight deep neural networks.
"""





import torch
from torch import nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class MobileNetv1(nn.Module):
    def __init__(self, width_multiplier=1, num_classes=1000):
        super().__init__()
        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(in_channels=3, out_channels=int(32 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(32 * alpha), out_channels=int(64 * alpha))
        )
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(64 * alpha), out_channels=int(128 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(128 * alpha), out_channels=int(128 * alpha))
        )
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(128 * alpha), out_channels=int(256 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(256 * alpha), out_channels=int(256 * alpha))
        )
        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(256 * alpha), out_channels=int(512 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
        )
        self.layer4 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(1024 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(1024 * alpha), out_channels=int(1024 * alpha))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(int(1024 * alpha), num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = MobileNetv1()
    y = model(x)
    print(y.shape)