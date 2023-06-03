""" 
PyTorch implementation of Searching for MobileNetV3

As described in https://arxiv.org/pdf/1905.02244

MobileNetV3 is tuned to mobile phone CPUs through a combination of hardwareaware 
network architecture search (NAS) complemented by the NetAdapt algorithm and then 
subsequently improved through novel architecture advances
"""


import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, expand_channels, out_channels,
                 kernel_size, stride, act="HS", use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.act = nn.Hardswish() if act == "HS" else nn.ReLU()

        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size,
                               stride=stride, padding=(kernel_size - 1) // 2, groups=expand_channels)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        
        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = stride == 1 and in_channels == out_channels
        self.se = SELayer(expand_channels, int(expand_channels // 4)) if use_se else None
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        if self.se is not None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += x
        return out
    
class MobileNetv3(nn.Module):
    def __init__(self, inverted_residual_setting, last_channels=1024, drop=0.2, num_classes=1000):
        super().__init__()
        features = []
        features.append(nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        ))
        for ic, ec, oc, ks, s, act, se in inverted_residual_setting:
            features.append(InvertedBottleneck(in_channels=ic, expand_channels=ec,
                                               out_channels=oc, kernel_size=ks, stride=s,
                                               act=act, use_se=se))
        lastconv_in_channels = inverted_residual_setting[-1][2]
        lastconv_out_channels = 6 * lastconv_in_channels
        features.append(nn.Sequential(
            nn.Conv2d(lastconv_in_channels, lastconv_out_channels, kernel_size=1),
            nn.BatchNorm2d(lastconv_out_channels),
            nn.Hardswish(inplace=True)
            ))
        self.featuers = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(lastconv_out_channels, last_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(drop, inplace=True),
            nn.Linear(last_channels, num_classes)
        )

    def forward(self, x):
        x = self.featuers(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

mobilenetv3_small_setting = [
    # input_channels, expand_channel, output_channel, kernel size, 
    # stride, activation, use_se
    [16, 16, 16, 3, 2, "RE", True],
    [16, 72, 24, 3, 2, "RE", False],
    [24, 88, 24, 3, 1, "RE", False],
    [24, 96, 40, 5, 2, "HS", True],
    [40, 240, 40, 5, 1, "HS", True],
    [40, 240, 40, 5, 1, "HS", True],
    [40, 120, 48, 5, 1, "HS", True],
    [48, 144, 48, 5, 1, "HS", True],
    [48, 288, 96, 5, 2, "HS", True],
    [96, 576, 96, 5, 1, "HS", True],
    [96, 576, 96, 5, 1, "HS", True]

]
mobilenetv3_large_setting = [
    [16, 16, 16, 3, 1, "RE", False],
    [16, 64, 24, 3, 2, "RE", False],
    [24, 72, 24, 3, 1, "RE", False],
    [24, 72, 40, 5, 2, "RE", True],
    [40, 120, 40, 5, 1, "RE", True],
    [40, 120, 40, 5, 1, "RE", True],
    [40, 240, 80, 3, 2, "HS", False],
    [80, 200, 80, 3, 1, "HS", False],
    [80, 184, 80, 3, 1, "HS", False],
    [80, 184, 80, 3, 1, "HS", False],
    [80, 480, 112, 3, 1, "HS", True],
    [112, 672, 112, 3, 1, "HS", True],
    [112, 672, 160, 5, 2, "HS", True],
    [160, 960, 160, 5, 1, "HS", True],
    [160, 960, 160, 5, 1, "HS", True],
]

def mobilenetv3_small(num_classes=1000):
    model = MobileNetv3(mobilenetv3_small_setting, last_channels=1024,
                        num_classes=num_classes)
    return model

def mobilenetv3_large(num_classes=1000):
    model = MobileNetv3(mobilenetv3_large_setting, last_channels=1280,
                        num_classes=num_classes)
    return model

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = mobilenetv3_small()
    y = model(x)
    print(y.shape)