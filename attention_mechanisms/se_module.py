
""" 
PyTorch implementation of Squeeze-and-Excitation Networks

As described in https://arxiv.org/pdf/1709.01507

The SE block is composed of two main components: the squeeze layer and the excitation layer. 
The squeeze layer reduces the spatial dimensions of the input feature maps by taking the average 
value of each channel. This reduces the number of parameters in the network, making it more efficient. 
The excitation layer then applies a learnable gating mechanism to the squeezed feature maps, which helps
to select the most informative channels and amplifies their contribution to the final output.

"""

import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32) #(B, C, H, W)
    attn = SELayer(channel=64, reduction=16)
    y = attn(x)
    print(y.shape)