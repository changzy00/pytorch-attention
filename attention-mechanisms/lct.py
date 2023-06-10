""" 
PyTorch implementation of Linear Context Transform Block

As described in https://arxiv.org/pdf/1909.03834v2

Linear Context Transform (LCT) block divides all channels into different groups
and normalize the globally aggregated context features within each channel group, 
reducing the disturbance from irrelevant channels. Through linear transform of 
the normalized context features, LCT models global context for each channel independently. 
"""




import torch
from torch import nn

class LCT(nn.Module):
    def __init__(self, channels, groups, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.avgpool(x).view(batch_size, self.groups, -1)
        mean = y.mean(dim=-1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        y_norm = self.w.reshape(1, -1, 1, 1) * y_norm + self.b.reshape(1, -1, 1, 1)
        y_norm = self.sigmoid(y_norm)
        return x * y_norm.expand_as(x)
        

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = LCT(64, 8)
    y = attn(x)
    print(y.shape)