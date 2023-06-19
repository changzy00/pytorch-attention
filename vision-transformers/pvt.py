""" 
PyTorch implementation of Pyramid vision transformer: A versatile backbone for dense prediction without convolutions

As described in https://arxiv.org/abs/2102.12122

Pyramid Vision Transformer (PVT), which overcomes the difficulties of porting Transformer to various dense prediction
tasks.
"""



import torch
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embedding_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size,
                              stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        H, W = H // self.patch_size, W // self.patch_size
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False,
                 attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim),
                nn.BatchNorm2d(dim)
            )

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, sr_ratio=1,
                 qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, sr_ratio, qkv_bias,
                              attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x
    
class PVT(nn.Module):
    def __init__(self, image_size=224, in_channels=3, num_classes=1000, mlp_ratios=None, 
                 depths=None, dims=None, num_heads=None, sr_ratios=None):
        super().__init__()
        self.patch_embedding_a = PatchEmbedding(
            image_size, patch_size=4, in_channels=in_channels, embedding_dim=dims[0]
        )
        self.patch_embedding_b = PatchEmbedding(
            image_size // 4, patch_size=2, in_channels=dims[0], embedding_dim=dims[1]
        )
        self.patch_embedding_c = PatchEmbedding(
            image_size // 8, patch_size=2, in_channels=dims[1], embedding_dim=dims[2]
        )
        self.patch_embedding_d = PatchEmbedding(
            image_size // 16, patch_size=2, in_channels=dims[2], embedding_dim=dims[3]
        )
        self.block_a = nn.Sequential(*[Block(dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0])
                                       for i in range(depths[0])])
        self.block_b = nn.Sequential(*[Block(dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1])
                                       for i in range(depths[1])])
        self.block_c = nn.Sequential(*[Block(dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2])
                                       for i in range(depths[2])])
        self.block_d = nn.Sequential(*[Block(dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3])
                                       for i in range(depths[3])])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embedding_a(x)
        for _, blk in enumerate(self.block_a):
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x, (H, W) = self.patch_embedding_b(x)
        for _, blk in enumerate(self.block_b):
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x, (H, W) = self.patch_embedding_c(x)
        for _, blk in enumerate(self.block_c):
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x, (H, W) = self.patch_embedding_d(x)
        for _, blk in enumerate(self.block_d):
            x = blk(x, H, W)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
    
def pvt_t(num_classes=1000):
    return PVT(mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 2, 2],
               dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
               sr_ratios=[8, 4, 2, 1], num_classes=num_classes)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = pvt_t()
    y = model(x)
    print(y.shape)
