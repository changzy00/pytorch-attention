""" 
PyTorch implementation of MetaFormer is Actually What You Need for Vision

As described in https://arxiv.org/abs/2111.11418

PoolFormer utilizes pooling as the basic token mixer to achieve excellent performance.
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
        
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chanels=3, 
                 dims=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chanels, dims, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = LayerNorm(dims, data_format="channels_first")
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x):
        return self.pool(x) - x

class Block(nn.Module):
    def __init__(self, dims, pool_size=3, mlp_ratio=4, drop=0):
        super().__init__()
        hidden_features = int(dims * mlp_ratio)
        self.norm1 = LayerNorm(dims, eps=1e-6, data_format="channels_first")
        self.pool = Pooling(pool_size)
        self.norm2 = LayerNorm(dims, eps=1e-6, data_format="channels_first")
        self.mlp = Mlp(dims, hidden_features, drop=drop)
    
    def forward(self, x):
        x = x + self.pool(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class PoolFormer(nn.Module):
    def __init__(self, layers=None, embedding_dims=None, mlp_ratios=None,
                 pool_size=3, num_classes=1000):
        super().__init__()
        self.downsamples = nn.ModuleList()
        stem = PatchEmbedding(patch_size=7, stride=4, padding=2, in_chanels=3, dims=embedding_dims[0])
        self.downsamples.append(stem)
        for i in range(3):
            downsample = PatchEmbedding(patch_size=3, stride=2, padding=1, 
                                        in_chanels=embedding_dims[i], dims=embedding_dims[i+1])
            self.downsamples.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(*[Block(embedding_dims[i], pool_size=pool_size, mlp_ratio=mlp_ratios[i])
                                    for j in range(layers[i])])
            self.stages.append(layer)
        self.head = nn.Linear(embedding_dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x

def poolformer_12(num_classes=1000):
    return PoolFormer(layers=[2, 2, 6, 2], embedding_dims=[64, 128, 320, 512],
                      mlp_ratios=[4, 4, 4, 4])
def poolformer_24(num_classes=1000):
    return PoolFormer(layers=[4, 4, 12, 4], embedding_dims=[64, 128, 320, 512],
                      mlp_ratios=[4, 4, 4, 4])

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = poolformer_12()
    y = model(x)
    print(y.shape)
