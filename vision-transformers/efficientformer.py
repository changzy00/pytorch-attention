""" 
PyTorch implementation of EfficientFormer: Vision Transformers at MobileNet Speed

As described in https://arxiv.org/abs/2212.08059

The proposed EfficientFormer complies with a dimension consistent design that smoothly leverages
hardware-friendly 4D MetaBlocks and powerful 3D MHSA blocks.
"""




import torch
from torch import nn

class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x):
        return self.pool(x) - x

class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.bn2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x
    
class MetaBlock4D(nn.Module):
    def __init__(self, dim, pool_size=3, exp=4):
        super().__init__()
        self.pool = Pooling(pool_size)
        hidden_featues = int(dim * exp)
        self.mlp = ConvMlp(dim, hidden_featues)
    
    def forward(self, x):
        x = x + self.pool(x)
        x = x + self.mlp(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, query_dim, num_heads, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.query_dim = query_dim
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qk = nn.Linear(dim, query_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.query_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    
class MetaBlock3D(nn.Module):
    def __init__(self, dim, query_dim, num_heads=8, exp=4, qkv_bias=False,
                 attn_drop=0, proj_drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, query_dim, num_heads, qkv_bias, 
                              attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * exp)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class EfficientFormer(nn.Module):
    def __init__(self, pool_size=3, stem_dim=24, embed_dim=[48, 96, 224, 448],
                 depths=[3, 2, 6, 1], transformer_depth=3, exp=4, attn_drop=0, 
                 proj_drop=0, qkv_bias=False, num_heads=8, query_dim=32,
                 num_classes=1000):
        super().__init__()
        stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(stem_dim, embed_dim[0], kernel_size=3, stride=2, padding=1)
        )
        self.downsamples = nn.ModuleList()
        self.downsamples.append(stem)
        for i in range(3):
            self.downsamples.append(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=3,
                          stride=2, padding=1)
            )
        self.stages = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(*[MetaBlock4D(embed_dim[i], pool_size, exp)
                                  for j in range(depths[i])])
            self.stages.append(layer)
        self.transformer = nn.Sequential(*[MetaBlock3D(embed_dim[-1], query_dim, num_heads,
                                                      exp, qkv_bias, attn_drop, proj_drop
                                                      ) for i in range(transformer_depth)])
        self.head = nn.Linear(embed_dim[-1], num_classes)
    
    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

def efficientformer_l1(num_classes=1000):
    return EfficientFormer(stem_dim=24, embed_dim=[48, 96, 224, 448],
                           depths=[3, 2, 6, 1], transformer_depth=3)

def efficientformer_l3(num_classes=1000):
    return EfficientFormer(stem_dim=32, embed_dim=[64, 128, 320, 512],
                           depths=[4, 4, 12, 3], transformer_depth=3)
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = efficientformer_l1()
    y = model(x)
    print(y.shape)

        