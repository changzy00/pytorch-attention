""" 
PyTorch implementation of CvT: Introducing Convolutions to Vision Transformers

As described in https://arxiv.org/abs/2103.15808

Convolutional vision Transformer (CvT), that improves Vision
Transformer (ViT) in performance and efficiency by introducing 
convolutions into ViT to yield the best of both designs. This 
is accomplished through two primary modifications: a hierarchy of 
Transformers containing a new convolutional token embedding, and 
a convolutional Transformer block leveraging a convolutional projection.
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
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, ks=3, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim  // num_heads
        self.scale = head_dim ** -0.5
        self.conv_proj_qkv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=ks, stride=1, padding=(ks - 1) // 2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, 3 * dim, 1)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.conv_proj_qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W)
        qkv = qkv.flatten(4).permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2).reshape(B, C, H * W)
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, kernel_size=3):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.attn = Attention(dim, num_heads, kernel_size)
        self.mlp = ConvMlp(dim, hidden_features)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(self.norm(x))
        return x
    
class CvT(nn.Module):
    def __init__(self, in_channels=3, layers=[1, 2, 10],
                 dims=[64, 192, 384], num_heads=[1, 3, 6],
                 num_classes=1000):
        super().__init__()
        self.downsamples = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=7, stride=4, padding=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsamples.append(stem)
        for i in range(2):
            self.downsamples.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first")
            ))

        self.stages = nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(*[ConvTransBlock(dims[i], num_heads[i])
                                    for j in range(layers[i])])
            self.stages.append(layer)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(3):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x

def cvt_13(num_classes=1000):
    return CvT(layers=[1, 3, 10], dims=[64, 192, 384], num_heads=[1, 3, 6])

def cvt_21(num_classes=1000):
    return CvT(layers=[1, 4, 16])

def cvt_w24(num_classes=1000):
    return CvT(layers=[2, 2, 20], dims=[192, 768, 1024], num_heads=[3, 12, 16])

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = cvt_13()
    y = model(x)
    print(y.shape)

