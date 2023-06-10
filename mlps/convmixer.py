""" 
PyTorch implementation of Patches Are All You Need?

As described in https://arxiv.org/pdf/2201.09792

ConvMixer, consists of a patch embedding layer followed by repeated applications
of a simple fully-convolutional block.
"""




import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x += shortcut
        x = self.pwconv(x)
        return x
    
class ConvMixer(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 in_channels=3,
                 kernel_size=9, 
                 patch_size=7,
                 num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(*[Block(dim, kernel_size)
                                      for i in range(depth)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = ConvMixer(128, 6)
    y = model(x)
    print(y.shape)