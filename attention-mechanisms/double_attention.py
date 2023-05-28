""" 
PyTorch implementation of A2-Nets: Double Attention Networks

As described in https://arxiv.org/pdf/1810.11579

The component is designed with a double attention mechanism in two steps, where the first step
gathers features from the entire space into a compact set through second-order
attention pooling and the second step adaptively selects and distributes features
to each location via another attention.
"""



import torch
from torch import nn
import torch.nn.functional as F


class DoubleAttention(nn.Module):
   
    def __init__(self, in_channels, c_m, c_n):
        
        super().__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.proj = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h, w = x.shape
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(b, self.c_m, h * w)
        attention_maps = B.view(b, self.c_n, h * w)
        attention_vectors = V.view(b, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        tmpZ = self.proj(tmpZ)
        return tmpZ

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = DoubleAttention(64, 32, 32)
    y = attn(x)
    print(y.shape)