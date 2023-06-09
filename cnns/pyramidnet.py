""" 
PyTorch implementation of Deep Pyramidal Residual Networks

As described in https://arxiv.org/pdf/1610.02915

The key idea is to concentrate on the feature map dimension by increasing it gradually instead of by
increasing it sharply at each residual unit with downsampling. In addition, network architecture works
as a mixture of both plain and residual networks by using zero-padded identity-mapping shortcut 
connections when increasing the feature map dimension.
"""

import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]
        if residual_channel != shortcut_channel:
            zero_padding = torch.autograd.Variable(torch.FloatTensor(batch_size, residual_channel - shortcut_channel,
                                                                     featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat([shortcut, zero_padding], dim=1)
        else:
            out += shortcut
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.bn4 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]
        if residual_channel != shortcut_channel:
            zero_padding = torch.autograd.Variable(torch.FloatTensor(batch_size, residual_channel - shortcut_channel,
                                                                     featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat([shortcut, zero_padding], dim=1)
        else:
            out += shortcut
        return out
    
class PyramidNet(nn.Module):
    def __init__(self, block, layers=[2, 2, 2, 2], alpha=48, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.growth_rate = alpha // sum(layers)
        self.input_features = self.inplanes
        self.conv1 = nn.Conv2d(3, self.input_features, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.input_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.features = self.input_features
        self.layer1 = self._make_layer(block, layers[0])
        self.layer2 = self._make_layer(block, layers[1], 2)
        self.layer3 = self._make_layer(block, layers[2], 2)
        self.layer4 = self._make_layer(block, layers[3], 2)

        self.output_features = self.input_features
        self.bn_final = nn.BatchNorm2d(self.output_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.output_features, num_classes)

    def _make_layer(self, block, depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.features = self.features + self.growth_rate
        layer = []
        layer.append(block(self.input_features, self.features, stride, downsample))
        for i in range(1, depth):
            temp_features = self.features + self.growth_rate
            layer.append(block(int(self.features * block.expansion), temp_features, 1))
            self.features = temp_features
        self.input_features = self.features * block.expansion
        return nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out

def pyramidnet18(num_classes=1000):
    return PyramidNet(BasicBlock, [2, 2, 2, 2], num_classes)

def pyramidnet34(num_classes=1000):
    return PyramidNet(BasicBlock, [3, 4, 6, 3], num_classes)

def pyramidnet50(num_classes=1000):
    return PyramidNet(Bottleneck, [3, 4, 6, 3], num_classes)

def pyramidnet101(num_classes=1000):
    return PyramidNet(Bottleneck, [3, 4, 23, 3], num_classes)

def pyramidnet152(num_classes=1000):
    return PyramidNet(Bottleneck, [3, 8, 36, 3], num_classes)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = pyramidnet18(num_classes=1000)
    y = model(x)
    print(y.shape)