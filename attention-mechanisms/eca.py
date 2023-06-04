""" 
PyTorch implementation of ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

As described in https://arxiv.org/abs/1910.03151

ECANet proposes a local crosschannel interaction strategy without dimensionality reduction, 
which can be efficiently implemented via 1D convolution.
"""




import torch
from torch import nn
import math

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = ECALayer(64)
    y = attn(x)
    print(y.shape)

