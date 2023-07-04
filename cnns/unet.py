""" 
PyTorch implementation of U-Net: Convolutional Networks for Biomedical
Image Segmentation

As described in https://arxiv.org/pdf/1505.04597.pdf

The architecture consists of a contracting path to capture
context and a symmetric expanding path that enables precise localization.
"""





import torch
from torch import nn

def conv_block(inplanes, planes):
    return nn.Sequential(
        nn.Conv2d(inplanes, planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(planes, planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # contracting path
        self.downconv_1 = conv_block(3, 64)
        self.downconv_2 = conv_block(64, 128)
        self.downconv_3 = conv_block(128, 256)
        self.downconv_4 = conv_block(256, 512)
        self.downconv_5 = conv_block(512, 1024)
        # expand path
        self.upscale_1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )
        self.upconv_1 = conv_block(1024, 512)

        self.upscale_2 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.upconv_2 = conv_block(512, 256)

        self.upscale_3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.upconv_3 = conv_block(256, 128)

        self.upscale_4 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.upconv_4 = conv_block(128, 64)

        self.out = nn.Conv2d(
            64, num_classes, 1
        )

    def forward(self, x):
        down_1 = self.downconv_1(x)
        down_2 = self.maxpool(down_1)
        down_3 = self.downconv_2(down_2)
        down_4 = self.maxpool(down_3)
        down_5 = self.downconv_3(down_4)
        down_6 = self.maxpool(down_5)
        down_7 = self.downconv_4(down_6)
        down_8 = self.maxpool(down_7)
        down_9 = self.downconv_5(down_8)

        up_1 = self.upscale_1(down_9)
        x = self.upconv_1(torch.cat([up_1, down_7], dim=1))
        up_2 = self.upscale_2(x)
        x = self.upconv_2(torch.cat([up_2, down_5], dim=1))
        up_3 = self.upscale_3(x)
        x = self.upconv_3(torch.cat([up_3, down_3], dim=1))
        up_4 = self.upscale_4(x)
        x = self.upconv_4(torch.cat([up_4, down_1], dim=1))
        x = self.out(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = Unet(10)
    y = model(x)
    print(y.shape)
