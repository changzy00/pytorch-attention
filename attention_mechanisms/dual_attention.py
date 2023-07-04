""" 
PyTorch implementation of Dual Attention Network for Scene Segmentation

As described in https://arxiv.org/pdf/1809.02983.pdf

Dual Attention Network (DANet) to adaptively integrate local features with their 
global dependencies based on the self-attention mechanism.
"""
import torch
from torch import nn

class PAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Conv2d(dim, dim, 1)
        self.c = nn.Conv2d(dim, dim, 1)
        self.d = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        n, c, h, w = x.shape
        B = self.b(x).flatten(2).transpose(1, 2)
        C = self.c(x).flatten(2)
        D = self.d(x).flatten(2).transpose(1, 2)
        attn = (B @ C).softmax(dim=-1)
        y = (attn @ D).transpose(1, 2).reshape(n, c, h, w)
        out = self.alpha * y + x
        return out

class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_ = x.flatten(2)
        attn = torch.matmul(x_, x_.transpose(1, 2))
        attn = attn.softmax(dim=-1)
        x_ = (attn @ x_).reshape(b, c, h, w)
        out = self.beta * x_ + x
        return out
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    #attn = PAM(64)
    attn = CAM()
    y = attn(x)
    print(y.shape)