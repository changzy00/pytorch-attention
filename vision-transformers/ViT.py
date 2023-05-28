""" 
PyTorch implementation of An image is worth 16x16 words: Transformers for image recognition at scale

As described in https://arxiv.org/pdf/2010.11929

Vision Transformer, is a computer vision model that uses a transformer architecture. 
It works by dividing an image into patches, which are then flattened and fed into a 
transformer encoder. The transformer encodes the sequence of patches and outputs a fixed-length 
vector representation of the image, which can be used for classification or other downstream tasks. 
By using a transformer, Vit is able to capture long-range dependencies between image patches and 
achieve state-of-the-art performance on several image classification benchmarks.
"""



import torch
from torch import nn
import math
from functools import partial
import torch.nn.functional as F

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
      
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
    
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

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
        x = self.gelu(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embedding_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        self.patch_size = patch_size

        self.num_patches = grid_size * grid_size
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.layernorm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features)
        self.layernorm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, 
                       patch_size=16, 
                       in_channels=3, 
                       depths=12,
                       num_heads=4,
                       mlp_ratio=4,
                       embedding_dim=768,
                       qkv_bias=False,
                       attn_drop=0,
                       proj_drop=0,
                       global_pool="token",
                       num_classes=1000
                    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels,
                                              embedding_dim)
        num_patches = self.patch_embedding.num_patches
        self.global_pool = global_pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches+1, embedding_dim))
        self.blocks = nn.Sequential(*[
            TransformerEncoder(embedding_dim, num_heads, mlp_ratio, qkv_bias,
                             attn_drop, proj_drop)
            for _ in range(depths)])
        self.head = nn.Linear(embedding_dim, num_classes)
        trunc_normal_(self.position_embedding, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.position_embedding.shape[1] - 1
        if npatch == N and w == h:
            return self.position_embedding
        class_pos_embed = self.position_embedding[:, 0]
        patch_pos_embed = self.position_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embedding.patch_size
        h0 = h // self.patch_embedding.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)
        x = torch.cat([x, self.cls_token.expand(x.shape[0], -1, -1)], dim=1)
        #x = x + self.position_embedding
        x = x + self.interpolate_pos_encoding(x, H, W)
        x = self.blocks(x)
        if self.global_pool == "token":
            x = x[:, 0]
        elif self.global_pool == "avg":
            x = x[:, 1:].mean(dim=1)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = VisionTransformer()
    y = model(x)
    print(y.shape)   # [2, 1000]


