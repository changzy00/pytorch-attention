""" 
PyTorch implementation of MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer

As described in https://arxiv.org/abs/2110.02178

MobileViT, a light-weight and general-purpose vision transformer for mobile devices. 
MobileViT presents a different perspective for the global processing of information with
transformers.
"""



import torch
from torch import nn

class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3,
                               stride=stride, padding=1, groups=in_channels * expand_ratio)
        self.bn2 = nn.BatchNorm2d(in_channels * expand_ratio)

        self.conv3 = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += x
        return out
    

class MVTBlock(nn.Module):
    def __init__(self, in_channels, dim, depth=2, kernel_size=3, patch_size=2):
        super().__init__()
        assert dim > in_channels
        self.patch_h = patch_size
        self.patch_w = patch_size
        self.dim = dim
        l = depth
        n = kernel_size
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=n, stride=1, padding=n//2),
            nn.Conv2d(in_channels, dim, 1)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4)
        self.global_rep = nn.TransformerEncoder(encoder_layer, num_layers=l)
        self.proj = nn.Conv2d(dim, in_channels, 1)
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=n, stride=1, padding=n//2)

    def forward(self, x):
        B, C, H, W = x.shape
        num_patch_h = H // self.patch_h
        num_patch_w = W // self.patch_w
        num_patches = num_patch_h * num_patch_w
        patch_area = self.patch_h * self.patch_w
        y = self.local_rep(x)
        y = y.reshape(B * self.dim * num_patch_h, self.patch_h, num_patch_w, self.patch_w)
        y = y.transpose(1, 2)
        y = y.reshape(B, self.dim, num_patches, patch_area)
        y = y.transpose(1, 3)
        y = y.reshape(B * patch_area, num_patches, self.dim)
        y = self.global_rep(y) # BP, N, d
        y = y.reshape(B, patch_area, num_patches, self.dim)
        y = y.transpose(1, 3) # B, d, N, P
        y = y.reshape(B * self.dim * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        y = y.transpose(1, 2)
        y = y.reshape(B, self.dim, H, W)
        y = self.proj(y)
        out = torch.cat([x, y], dim=1)
        out = self.fuse(out)
        return out
    
class MobileViT(nn.Module):
    def __init__(self, stem_channel=16, channels=[24, 48, 64, 80], d=[60, 80, 96],
                 kernel_size=3, patch_size=2, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            MV2Block(16, stem_channel, stride=1)
        )
        self.stage1 = nn.Sequential(
            MV2Block(stem_channel, channels[0], stride=2),
            MV2Block(channels[0], channels[0], stride=1),
            MV2Block(channels[0], channels[0], stride=1)
        )
        self.stage2 = nn.Sequential(
            MV2Block(channels[0], channels[1], stride=2),
            MVTBlock(channels[1], d[0], depth=2)
        )
        self.stage3 = nn.Sequential(
            MV2Block(channels[1], channels[2], stride=2),
            MVTBlock(channels[2], d[1], depth=4)
        )
        self.stage4 = nn.Sequential(
            MV2Block(channels[2], channels[3], stride=2),
            MVTBlock(channels[3], d[2], depth=3)
        )
        self.proj = nn.Conv2d(channels[-1], channels[-1] * 4, 1)
        self.head = nn.Linear(channels[-1] * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.proj(x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x

def mobilevit_xxs(num_classes=1000):
    return MobileViT(stem_channel=16, channels=[24, 48, 64, 80],
                     d=[60, 80, 96])   

def mobilevit_xs(num_classes=1000):
    return MobileViT(stem_channel=32, channels=[48, 64, 80, 96],
                     d=[96, 120, 144]) 

def mobilevit_s(num_classes=1000):
    return MobileViT(stem_channel=32, channels=[64, 96, 128, 160],
                     d=[144, 192, 240]) 
if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    model = mobilevit_xxs()
    y = model(x)
    print(y.shape)




