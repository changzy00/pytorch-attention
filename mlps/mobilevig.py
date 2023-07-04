""" 
PyTorch implementation of MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications

As described in https://arxiv.org/pdf/2307.00395.pdf

The authors propose a novel mobile CNN-GNN architecture
for vision tasks using our proposed SVGA, maxrelative graph convolution, 
and concepts from mobile CNN and mobile vision transformer architectures.
"""





import torch
from torch import nn

class MBConvBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.bn1 = nn.BatchNorm2d(dim * expand_ratio)
        self.conv2 = nn.Conv2d(dim * expand_ratio, dim * expand_ratio, kernel_size=3,
                               stride=1, padding=1, groups=dim * expand_ratio)
        self.bn2 = nn.BatchNorm2d(dim * expand_ratio)
        self.conv3 = nn.Conv2d(dim * expand_ratio, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        shortcut = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) + shortcut
        return x 

class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
            
        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)
        
class Grapher(nn.Module):
    def __init__(self, dim, k=2):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )
        self.graph_conv = MRConv4d(dim, dim * 2, k)
        self.fc2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x += shortcut
        return x
    
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SVGABlock(nn.Module):
    def __init__(self, dim, k=2):
        super().__init__()
        self.grapher = Grapher(dim, k)
        self.ffn = FFN(dim, dim * 4)
    
    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)
        return x


class MobileViG(nn.Module):
    def __init__(self, embed_dims=[42, 84, 168, 256], depths=[2, 2, 6, 2],
                 k=2, num_classes=1000):
        super().__init__()
        self.downsamples = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        self.downsamples.append(stem)
        for i in range(3):
            downsample = nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dims[i+1])
            )
            self.downsamples.append(downsample)
        
        self.stages = nn.ModuleList()
        for i in range(4):
            if i == 3:
                layer = nn.Sequential(*[SVGABlock(embed_dims[i], k) for j in range(depths[i])])
                self.stages.append(layer)
            else:
                layer = nn.Sequential(*[MBConvBlock(embed_dims[i]) for j in range(depths[i])])
                self.stages.append(layer)
        self.head = nn.Linear(embed_dims[-1], num_classes)
    
    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x

def mobilevig_ti(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 168, 256], depths=[2, 2, 6, 2], k=2, num_classes=num_classes)

def mobilevig_s(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 176, 256], depths=[3, 3, 9, 3], k=2, num_classes=num_classes)

def mobilevig_m(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 224, 400], depths=[3, 3, 9, 3], k=2, num_classes=num_classes)

def mobilevig_b(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 240, 464], depths=[5, 5, 15, 5], k=2, num_classes=num_classes)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = mobilevig_ti(num_classes=1000)
    y = model(x)
    print(y.shape)