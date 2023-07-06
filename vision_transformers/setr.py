""" 
PyTorch implementation of Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective
with Transformers

As described in https://arxiv.org/abs/2012.15840

We deploy a pure transformer (i.e., without convolution and
resolution reduction) to encode an image as a sequence of
patches. With the global context modeled in every layer of
the transformer, this encoder can be combined with a simple
decoder to provide a powerful segmentation model, termed
SEgmentation TRansformer (SETR).
"""



import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, in_channel=3, patch_size=16, embed_dim=768):
        super().__init__()
        grid_size = image_size // patch_size
        self.num_patches = grid_size ** 2
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_heads=  num_heads
        self.qkv = nn.Linear(dim, 3 * dim, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = x.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, mlp_ratio=4, 
                 attn_drop=0, proj_drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, image_size=224, in_channel=3, patch_size=16, embed_dim=768,
                 num_heads=8, mlp_ratio=4, qkv_bias=False, depths=24, attn_drop=0, proj_drop=0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(image_size, in_channel,
                                              patch_size, embed_dim)
        num_patches = self.patch_embedding.num_patches
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, qkv_bias,
                                            mlp_ratio, attn_drop, proj_drop) for i in range(depths)])
    
    def forward(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embedding(x)
        x += self.position_embedding
        x = self.blocks(x)
        x = x.reshape(B, -1, H, W)
        return x

class SETR(nn.Module):
    def __init__(self, image_size=480, in_channel=3, patch_size=16, embed_dim=1024,
                 num_heads=8, mlp_ratio=4, qkv_bias=False, depths=12, attn_drop=0, 
                 proj_drop=0, num_classes=19):
        super().__init__()
        self.encoder = TransformerEncoder(image_size, in_channel, patch_size, embed_dim,
                                          num_heads, mlp_ratio, qkv_bias, depths, attn_drop, proj_drop)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, 1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=H, mode="bilinear", align_corners=True)
        return x

x = torch.randn(2, 3, 480, 480)
model = SETR(num_classes=19)
y = model(x)
print(y.shape)
