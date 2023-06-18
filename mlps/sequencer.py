
""" 
PyTorch implementation of Sequencer: Deep LSTM for Image Classification

As described in https://arxiv.org/abs/2205.01972

Sequencer architecture takes nonoverlapping patches as input and projects them onto the feature map. 
Sequencer block, which is a core component of Sequencer, consists of the following sub-components: 
(1) BiLSTM layer can mix spatial information more memory-economically for high-resolution images than 
Transformer layer and more globally than CNN. (2) MLP for channel-mixing.
"""



import torch
from torch import nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class Mlp(nn.Module):
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
           
class BiLSTM2D(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.lstm_v = nn.LSTM(in_features, hidden_features, num_layers=1, 
                              batch_first=True, bias=True, bidirectional=True)
        self.lstm_h = nn.LSTM(in_features, hidden_features, num_layers=1, 
                              batch_first=True, bias=True, bidirectional=True)
        self.fc = nn.Conv2d(4 * hidden_features, in_features, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h, _ = self.lstm_h(x.permute(0, 3, 2, 1).reshape(-1, H, C))
        h = h.reshape(B, W, H, -1).transpose(1, 3)
        w, _ = self.lstm_h(x.permute(0, 2, 3, 1).reshape(-1, W, C))
        w = w.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = torch.cat([h, w], dim=1)
        x = self.fc(x)
        return x

class Block(nn.Module):
    def __init__(self, in_features, hidden_features, mlp_ratio=4):
        super().__init__()
        self.norm1 = LayerNorm(in_features, data_format="channels_first")
        self.bilstm = BiLSTM2D(in_features, hidden_features)
        self.norm2 = LayerNorm(in_features, data_format="channels_first")
        self.mlp = Mlp(in_features, in_features * mlp_ratio)
    
    def forward(self, x):
        x = x + self.bilstm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=7, in_channels=3, embedding_dim=768):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size,
                              stride=patch_size)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Sequencer(nn.Module):
    def __init__(self, depths=[4, 3, 8, 3], embed_dim=[192, 384, 384, 384],
                 hidden_dim=[48, 96, 96, 96], patch_size=[7, 2, 1, 1],
                 mlp_ratios=[3, 3, 3, 3], num_classes=1000):
        super().__init__()
        self.downsamples = nn.ModuleList()
        self.downsamples.append(
            PatchEmbedding(patch_size[0], 3, embed_dim[0])
        )
        for i in range(3):
            self.downsamples.append(PatchEmbedding(patch_size[i+1], embed_dim[i], embed_dim[i+1]))
        self.stages = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(*[Block(embed_dim[i], hidden_dim[i], mlp_ratios[i])
                                    for j in range(depths[i])])
            self.stages.append(layer)
        self.head = nn.Linear(embed_dim[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-1, -2])
        x = self.head(x)
        return x
    
def sequencer_s(num_classes=1000):
    return Sequencer(depths=[4, 3, 8, 3], embed_dim=[192, 384, 384, 384],
                     hidden_dim=[48, 96, 96, 96], patch_size=[7, 2, 1, 1],
                     num_classes=num_classes)

def sequencer_m(num_classes=1000):
    return Sequencer(depths=[4, 3, 14, 3], embed_dim=[192, 384, 384, 384],
                     hidden_dim=[48, 96, 96, 96], patch_size=[7, 2, 1, 1],
                     num_classes=num_classes)
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = sequencer_s()
    y = model(x)
    print(y.shape)