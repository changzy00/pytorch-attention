""" 
PyTorch implementation of When Shift Operation Meets Vision Transformer:
An Extremely Simple Alternative to Attention Mechanism

As described in https://arxiv.org/abs/2201.10801

A small portion of channels will be shifted along
4 spatial directions, namely left, right, top, and down, while
the remaining channels keep unchanged. After shifting, the
out-of-scope pixels are simply dropped and the vacant pixels
are zero padded.
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
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, gamma=1/12, mlp_ratio=4, drop=0, step=1):
        super().__init__()
        self.gamma = gamma
        self.step = step
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.mlp = Mlp(dim, dim * mlp_ratio)

    def shift_feature(self, x, gamma, step):
        B, C, H, W = x.shape
        g = int(C * gamma)
        y = torch.zeros_like(x)
        y[:, g * 0:g * 1, :-step, :] = x[:, g * 0:g * 1, step:, :] # shift up
        y[:, g * 1:g * 2, step:, :] = x[:, g * 1:g * 2, :-step, :] # shift down
        y[:, g * 2:g * 3, :, :-step] = x[:, g * 2:g * 3, :, step:] # shift left
        y[:, g * 3:g * 4,:, step:] = x[:, g * 3:g * 4, :, :-step] # shift right
        y[:, g * 4:, :, :] = x[:, g * 4:, :, :]
        return y
    
    def forward(self, x):
        x = self.shift_feature(x, self.gamma, self.step)
        x = x + self.mlp(self.norm(x))
        return x

class ShiftViT(nn.Module):
    def __init__(self, embed_dims=[96, 192, 384, 768], gamma=1/12, 
                 mlp_ratio=4, drop=0, step=1, depths=[6, 8, 18, 6],
                 num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 48, kernel_size=4, stride=4)
        self.downsamples = nn.ModuleList()
        self.downsamples.append(
                    nn.Conv2d(48, embed_dims[0], kernel_size=1)
                )
        for i in range(3):
            self.downsamples.append(
                nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2)
            )
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(
                nn.Sequential(*[Block(embed_dims[i], gamma, mlp_ratio, drop, step)
                                for j in range(depths[i])])
            )
        self.head = nn.Linear(embed_dims[-1], num_classes)
    def forward(self, x):
        x = self.stem(x)
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x

def shift_t(num_classes):
    return ShiftViT(embed_dims=[96, 192, 384, 768], depths=[6, 8, 18, 6], gamma=1/12, num_classes=num_classes)

def shift_s(num_classes):
    return ShiftViT(embed_dims=[96, 192, 384, 768], depths=[10, 18, 36, 10], gamma=1/12, num_classes=num_classes)

def shift_b(num_classes):
    return ShiftViT(embed_dims=[128, 256, 512, 1024], depths=[10, 18, 36, 10], gamma=1/16, num_classes=num_classes)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = shift_t(num_classes=1000)
    y = model(x)
    print(y.shape)