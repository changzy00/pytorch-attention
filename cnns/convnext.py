""" 
PyTorch implementation of A ConvNet for the 2020s

As described in https://arxiv.org/abs/2201.03545

ConvNeXts, a pure ConvNet model that can compete favorably with state-of-the-art
hierarchical vision Transformers across multiple computer vision benchmarks, while retaining
the simplicity and efficiency of standard ConvNets.
"""



import torch
from torch import nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.norm(x).permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut
    
class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, layers=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], num_classes=1000):
        super().__init__()
        self.downsample = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample.append(stem)
        for i in range(3):
            self.downsample.append(
                nn.Sequential(
                   LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                   nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )
            )
        self.stages = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(*[Block(dims[i]) for _ in range(layers[i])])
            self.stages.append(layer)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.fc = nn.Linear(dims[-1], num_classes)
    
    def forward(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-1, -2])
        x = self.norm(x)
        x = self.fc(x)
        return x

def convnext_tiny(num_classes):
    return ConvNeXt(layers=[3, 3, 9, 3], dims=[96, 192, 384, 768])

def convnext_small(num_classes):
    return ConvNeXt(layers=[3, 3, 27, 3], dims=[96, 192, 384, 768])

def convnext_small(num_classes):
    return ConvNeXt(layers=[3, 3, 27, 3], dims=[128, 256, 512, 1024])

def convnext_large(num_classes):
    return ConvNeXt(layers=[3, 3, 27, 3], dims=[192, 384, 768, 1536])

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = ConvNeXt()
    y = model(x)
    print(y.shape)

        

