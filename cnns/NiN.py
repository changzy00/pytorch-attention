""" 
PyTorch implementation of Network In Network

As described in https://arxiv.org/pdf/1312.4400v3

The author build micro neural networks with more complex structures to abstract 
the data within the receptive field. They instantiate the micro neural network with 
a multilayer perceptron, which is a potent function approximator. The feature maps 
are obtained by sliding the micro networks over the input in a similar manner as CNN; 
they are then fed into the next layer.
"""





import torch
from torch import nn

class NiN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            self.mlpconv_layer(3, 96, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.mlpconv_layer(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.mlpconv_layer(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.mlpconv_layer(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def mlpconv_layer(self, inplanes, planes, kernel_size, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 1),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.net(x).flatten(1)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = NiN()
    y = model(x)
    print(y.shape)
