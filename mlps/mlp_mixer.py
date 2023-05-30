""" 
PyTorch implementation of MLP-Mixer: An all-MLP Architecture for Vision

As described in https://arxiv.org/pdf/2105.01601

MLP-Mixer contains two types of layers: one with MLPs applied independently to
image patches (i.e. “mixing” the per-location features), and one with MLPs applied
across patches (i.e. “mixing” spatial information).
"""



import torch
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, embedding_dim, sequence_len, mlp_ratio=[0.5, 4], drop=0):
        super().__init__()
        token_dim = int(embedding_dim * mlp_ratio[0])
        channel_dim = int(embedding_dim * mlp_ratio[1])
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.token_mlp = Mlp(sequence_len, token_dim, drop=drop)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.channel_mlp = Mlp(embedding_dim, channel_dim, drop=drop)

    def forward(self, x):
        # token-mixing mlp
        x = x + self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        # channel-mixing mlp
        x = x + self.channel_mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_chanels=3, embedding_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        self.num_patches = grid_size * grid_size
        self.proj = nn.Conv2d(in_chanels, embedding_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, dim=512, depth=12, image_size=224, patch_size=16, 
                 in_channels=3, drop=0, num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.blocks = nn.Sequential(*[MixerLayer(dim, self.patch_embedding.num_patches, drop=drop)
                                      for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = MLP_Mixer()
    y = model(x)
    print(y.shape)
