""" 
PyTorch implementation of MOAT: Alternating Mobile Convolution and Attention Brings Strong Vision Models

As described in https://arxiv.org/pdf/2210.01820.pdf

The proposed MOAT block modifies the Transformer
block by first replacing its MLP with a MBConv block, and then reversing the order of attention
and MBConv. The replacement of MLP with MBConv brings more representation capacity to the
network, and reversing the order (MBConv comes before self-attention) delegates the downsampling
duty to the strided depthwise convolution within the MBConv, learning a better downsampling kernel.
"""



import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class MBConv(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expand_ratio=4, use_se=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes * expand_ratio, 1)
        self.bn1 = nn.BatchNorm2d(inplanes * expand_ratio)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(inplanes * expand_ratio, inplanes * expand_ratio, kernel_size=3,
                               stride=stride, padding=1, groups=inplanes * expand_ratio)
        self.bn2 = nn.BatchNorm2d(inplanes * expand_ratio)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(inplanes * expand_ratio, 4)
        self.conv3 = nn.Conv2d(inplanes * expand_ratio, planes, 1)
        self.shortcut = stride == 1 and inplanes == planes

    def forward(self, x):
        out = self.bn(x)
        out = self.act(self.bn1(self.conv1(out)))
        out = self.act(self.bn2(self.conv2(out)))
        if self.use_se:
            out = self.se(out)
        out = self.conv3(out)
        if self.shortcut:
            out += x
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
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

class MOATBlock(nn.Module):
    def __init__(self, inplanes, planes, num_heads=8, stride=1, qkv_bias=False, 
                 expand_ratio=4, use_se=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.mbconv = MBConv(inplanes, planes, stride, expand_ratio, use_se)
        self.layernorm = nn.LayerNorm(planes)
        self.attn = Attention(planes, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, x):
        x = self.mbconv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.attn(self.layernorm(x))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
    
class MOAT(nn.Module):
    def __init__(self, stem_dim=64, embed_dims=[96, 192, 384, 768],
                 depths=[2, 3, 7, 2], num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, stem_dim, kernel_size=3, stride=2, padding=1)
        inplanes = stem_dim
        layer1 = []
        for i in range(depths[0]):
            stride = 2 if i == 0 else 1
            layer1.append(MBConv(inplanes, embed_dims[0], stride, use_se=True))
            inplanes = embed_dims[0]
        self.layer1 = nn.Sequential(*layer1)
        layer2 = []
        for i in range(depths[1]):
            stride = 2 if i == 0 else 1
            layer2.append(MBConv(inplanes, embed_dims[1], stride, use_se=True))
            inplanes = embed_dims[1]
        self.layer2 = nn.Sequential(*layer2)
        layer3 = []
        for i in range(depths[2]):
            stride = 2 if i == 0 else 1
            layer3.append(MOATBlock(inplanes, embed_dims[2], stride=stride, use_se=False))
            inplanes = embed_dims[2]
        self.layer3 = nn.Sequential(*layer3)
        layer4 = []
        for i in range(depths[3]):
            stride = 2 if i == 0 else 1
            layer4.append(MOATBlock(inplanes, embed_dims[3], stride=stride, use_se=False))
            inplanes = embed_dims[3]
        self.layer4 = nn.Sequential(*layer4)
        self.head = nn.Linear(inplanes, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x

def moat_0(num_classes=1000):
    return MOAT(stem_dim=64, embed_dims=[96, 192, 384, 768], depths=[2, 3, 7, 2],
                num_classes=num_classes)

def moat_1(num_classes=1000):
    return MOAT(stem_dim=64, embed_dims=[96, 192, 384, 768], depths=[2, 6, 14, 2],
                num_classes=num_classes)

def moat_2(num_classes=1000):
    return MOAT(stem_dim=128, embed_dims=[128, 256, 512, 1024], depths=[2, 6, 14, 2],
                num_classes=num_classes)

def moat_3(num_classes=1000):
    return MOAT(stem_dim=160, embed_dims=[160, 320, 640, 1280], depths=[2, 12, 28, 2],
                num_classes=num_classes)

def moat_4(num_classes=1000):
    return MOAT(stem_dim=256, embed_dims=[256, 512, 1024, 2048], depths=[2, 12, 28, 2],
                num_classes=num_classes)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = moat_0()
    y = model(x)
    print(y.shape)
