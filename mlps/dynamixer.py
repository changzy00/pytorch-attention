""" 
PyTorch implementation of DynaMixer: A Vision MLP Architecture with Dynamic Mixing

As described in https://arxiv.org/pdf/2201.12083

DynaMixer consists of a patch embedding layer, several mixer layers, a global
average pooling layer, and a classifier head. The patch embedding layer transforms 
input non-overlapping patches into corresponding input tokens, which are
fed into a sequence of mixer layers to generate the output tokens. All output
tokens are averaged in the average pooling layer, and the final prediction is
generated with a classifier head. The mixer layer (middle part) contains two
layer-normalization layers, a DynaMixer block and a channel-MLP block. The
DynaMixer block (right part) performs row mixing and column mixing via the
DynaMixer operations, and a simple channel-mixing via linear projection. The mixing 
results are summed and outputted with a linear transformation.
"""



import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embedding_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        self.num_patches = grid_size * grid_size
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size,
                              stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynaMixerOp(nn.Module):
    def __init__(self, dim, sequence_len, num_heads, reduced_dim=2):
        super().__init__()
        self.num_heads = num_heads
        self.reduced_dim = reduced_dim
        self.compress = nn.Linear(dim, num_heads * reduced_dim)
        self.generate  = nn.Linear(sequence_len * reduced_dim, sequence_len * sequence_len)
        self.proj  = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        weight = self.compress(x).reshape(B, N, self.num_heads, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_heads, -1)
        weight = self.generate(weight).reshape(B, self.num_heads, N, N)
        weight = weight.softmax(dim=-1)
        x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (weight @ x).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, sequence_len=32, num_heads=8, reduced_dim=2, 
                 qkv_bias=False, proj_drop=0):
        super().__init__()
        self.h_mix = DynaMixerOp(dim, sequence_len, num_heads, reduced_dim)
        self.w_mix = DynaMixerOp(dim, sequence_len, num_heads, reduced_dim)
        self.c_mix = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.h_mix(x.permute(0, 2, 1, 3).reshape(-1, H, C)).reshape(B, W, H, C).transpose(1, 2)
        w = self.w_mix(x.reshape(-1, W, C)).reshape(B, H, W, C)
        c = self.c_mix(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(dim=2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class DynaMixer(nn.Module):
    def __init__(self, dim=64, num_heads=8, reduced_dim=2, depths=12, image_szie=224,
                 patch_size=7, in_channels=3, drop=0, num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_szie, patch_size, in_channels, dim)
        resolution = int(self.patch_embedding.num_patches ** 0.5)
        self.blocks = nn.Sequential(*[Block(dim, resolution, num_heads, reduced_dim)
                                      for i in range(depths)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x).permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2).flatten(2).mean(dim=2)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = DynaMixer()
    y = model(x)
    print(y.shape)


