""" 
PyTorch implementation of Pay Attention to MLPs

As described in https://arxiv.org/pdf/2105.08050

gMLP, based on MLPs with gating, and show that it can perform as well as Transformers 
in key language and vision applications.
"""



import torch
from torch import nn

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, sequence_len):
        super().__init__()
        gate_dim = dim // 2
        self.norm = nn.LayerNorm(gate_dim)
        self.proj = nn.Linear(sequence_len, sequence_len)
    
    def init_weights(self):
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.proj(self.norm(v).transpose(-1, -2))
        return u * v.transpose(-1, -2)
    
class Block(nn.Module):
    def __init__(self, dim, sequence_len, mlp_ratio=4, drop=0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        channel_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, channel_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop, inplace=True)
        self.sgu = SpatialGatingUnit(channel_dim, sequence_len)
        self.fc2 = nn.Linear(channel_dim // 2, dim)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.sgu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embedding_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        self.num_patches = grid_size * grid_size
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(-1, -2)
        return x
    
class gMLP(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, dim=768,
                 mlp_ratio=4, drop=0, depths=12, num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size,
                                              in_channels, dim)
        self.blocks = nn.Sequential(*[Block(dim, self.patch_embedding.num_patches, 
                                            mlp_ratio, drop) for i in range(depths)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = gMLP()
    y = model(x)
    print(y.shape)