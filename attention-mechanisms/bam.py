""" 
PyTorch implementation of Bam: Bottleneck attention module

As described in http://bmvc2018.org/contents/papers/0092.pdf

Given a 3D feature map, BAM produces a 3D attention map to emphasize important elements. BAM
decomposes the process of inferring a 3D attention map in two streams , so that the
computational and parametric overhead are significantly reduced. As the channels of feature
maps can be regarded as feature detectors, the two branches (spatial and channel) explicitly
learn "what" and "where" to focus on.
"""




import torch
from torch import nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.bn = nn.BatchNorm1d(channel)
    
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.mlp(y)
        y = self.bn(y).view(b, c, 1, 1)
        return y.expand_as(x)
    
class SpatialGate(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3, dilation_val=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(y)
        return y.expand_as(x)
    
class BAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = ChannelGate(channel)
        self.spatial_attn = SpatialGate(channel)
        
    def forward(self, x):
        attn = F.sigmoid(self.channel_attn(x) + self.spatial_attn(x))
        return x + x * attn
        
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = BAM(64)
    y = attn(x)
    print(y.shape)
