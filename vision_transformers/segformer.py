""" 
PyTorch implementation of SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

As described in https://arxiv.org/abs/2105.15203

SegFormer, a simple, efficient yet powerful semantic segmentation
framework which unifies Transformers with lightweight multilayer perceptron
(MLP) decoders.
"""




import torch
from torch import nn
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                                groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        return x
class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False,
                 attn_drop=0, proj_drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FFN(dim, dim * mlp_ratio)
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=7, stride=4,
                 in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size//2)
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x, (H, W)

        
class MiT(nn.Module):
    def __init__(self, image_size=224, num_classes=1000, patch_size=7, in_channels=3,
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.patch_embedding1 = PatchEmbedding(image_size, patch_size=7, stride=4, in_channels=in_channels,
                                               embed_dim=embed_dims[0])
        self.patch_embedding2 = PatchEmbedding(image_size // 4, patch_size=3, stride=2, in_channels=embed_dims[0],
                                               embed_dim=embed_dims[1])
        self.patch_embedding3 = PatchEmbedding(image_size // 8, patch_size=3, stride=2, in_channels=embed_dims[1],
                                               embed_dim=embed_dims[2])
        self.patch_embedding4 = PatchEmbedding(image_size // 16, patch_size=3, stride=2, in_channels=embed_dims[2],
                                               embed_dim=embed_dims[3])
        
        self.block_1 = nn.Sequential(*[Block(embed_dims[0], num_heads[0], mlp_ratios[0],
                                             sr_ratios[0])
                                             for i in range(depths[0])])
        self.block_2 = nn.Sequential(*[Block(embed_dims[1], num_heads[1], mlp_ratios[1],
                                             sr_ratios[1])
                                             for i in range(depths[1])])
        self.block_3 = nn.Sequential(*[Block(embed_dims[2], num_heads[2], mlp_ratios[2],
                                             sr_ratios[2])
                                             for i in range(depths[2])])
        self.block_4 = nn.Sequential(*[Block(embed_dims[3], num_heads[3], mlp_ratios[3],
                                             sr_ratios[3])
                                             for i in range(depths[3])])
    
    def forward(self, x):
        B = x.shape[0]
        outputs = []
        x, (H, W) = self.patch_embedding1(x)
        for _, blk in enumerate(self.block_1):
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outputs.append(x)
        x, (H, W) = self.patch_embedding2(x)
        for _, blk in enumerate(self.block_2):
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outputs.append(x)
        x, (H, W) = self.patch_embedding3(x)
        for _, blk in enumerate(self.block_3):
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outputs.append(x)
        x, (H, W) = self.patch_embedding4(x)
        for _, blk in enumerate(self.block_4):
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outputs.append(x)
        return outputs
    
def mit_b0():
    return MiT(embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
               mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2], sr_ratios=[8, 2, 4, 1])

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class SegFormer(nn.Module):
    def __init__(self, num_classes=10, in_dims=[32, 64, 160, 256], decoder_dim=256):
        super().__init__()
        self.encoder = mit_b0()
        self.decoder_dim = decoder_dim
        self.mlp_a = MLP(in_dims[0], decoder_dim)
        self.mlp_b = MLP(in_dims[1], decoder_dim)
        self.mlp_c = MLP(in_dims[2], decoder_dim)
        self.mlp_d = MLP(in_dims[3], decoder_dim)
        self.fuse = nn.Conv2d(decoder_dim * 4, decoder_dim, 1)
        self.out = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        x1, x2, x3, x4 = features
        B, _, H, W = x1.shape
        x1_ = self.mlp_a(x1).transpose(1, 2).reshape(B, -1, x1.shape[2], x1.shape[3])
        x2_ = self.mlp_b(x2).transpose(1, 2).reshape(B, -1, x2.shape[2], x2.shape[3])
        x2_ = F.interpolate(x2_, size=(H, W), mode="bilinear")
        x3_ = self.mlp_c(x3).transpose(1, 2).reshape(B, -1, x3.shape[2], x3.shape[3])
        x3_ = F.interpolate(x3_, size=(H, W), mode="bilinear")
        x4_ = self.mlp_d(x4).transpose(1, 2).reshape(B, -1, x4.shape[2], x4.shape[3])
        x4_ = F.interpolate(x4_, size=(H, W), mode="bilinear")
        x = self.fuse(torch.cat([x1_, x2_, x3_, x4_], dim=1))
        x = self.out(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = SegFormer(num_classes=50)
    y = model(x)
    print(y.shape)
        
        