""" 
PyTorch implementation of Selective Kernel Networks

As described in https://arxiv.org/abs/1903.06586

A building block called Selective Kernel (SK) unit is designed, in which multiple 
branches with different kernel sizes are fused using softmax attention that is guided 
by the information in these branches. Different attentions on these branches yield 
different sizes of the effective receptive fields of neurons in the fusion layer.
"""



import torch
from torch import nn

class SKLayer(nn.Module):
    def __init__(self, inplanes, planes, groups=32, ratio=16):
        super().__init__()
        d = max(planes // ratio, 32)
        self.planes = planes
        self.split_3x3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.split_5x5 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=2, dilation=2, groups=groups),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, d),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)

    def forward(self, x):
        batch_size = x.shape[0]
        u1 = self.split_3x3(x)
        u2 = self.split_5x5(x)
        u = u1 + u2
        s = self.avgpool(u).flatten(1)
        z = self.fc(s)
        attn_scores = torch.cat([self.fc1(z), self.fc2(z)], dim=1)
        attn_scores = attn_scores.view(batch_size, 2, self.planes)
        attn_scores = attn_scores.softmax(dim=1)
        a = attn_scores[:,0].view(batch_size, self.planes, 1, 1)
        b = attn_scores[:,1].view(batch_size, self.planes, 1, 1)
        u1 = u1 * a.expand_as(u1)
        u2 = u2 * b.expand_as(u2)
        x = u1 + u2
        return x

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = SKLayer(64, 64)
    y = attn(x)
    print(y.shape)