import torch
import torch.nn as nn



"""
Learned positional embedding for 2D spatiotemporal grids.

- Input: tensor with shape [B, T, S]
- Adds a trainable positional encoding with shape [T, S, d_model]
- Uses LayerNorm to stabilize the result
"""

class PositionalEmbeddingCONV(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbeddingCONV, self).__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, S = x.shape

        pos_emb = nn.Parameter(torch.randn(T, S, self.d_model))


        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)
        pos_emb = pos_emb.to(x.device)

        x = x.unsqueeze(-1) + pos_emb

        x = self.norm(x)
        x = x.to(self.norm.weight.device)
        return x



"""
Token (value) embedding using a 2D convolutional neural network.

- Applies multiple Conv2D + ReLU layers over [l x k] spatial patches
- Captures local spatial structure from raw values
- Output shape: [B, T, S, D]
"""


class TokenEmbeddingCONV(nn.Module):
    def __init__(self, c_in, d_model,l,k):
        super(TokenEmbeddingCONV, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.s1 = l
        self.s2 = k
        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 初始化卷积层参数
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def debug_tensor(self, tensor, name="Tensor"):
        print(f"Debugging {name}:")
        print(f"Shape: {tensor.shape}")
        if tensor.numel() > 0:  # 确保张量有元素
            print(f"Max: {tensor.max().item()}")
            print(f"Min: {tensor.min().item()}")
            print(f"Mean: {tensor.mean().item()}")
            print(f"Contains NaN: {torch.isnan(tensor).any().item()}")
            print(f"Contains Inf: {torch.isinf(tensor).any().item()}")
        else:
            print("Empty tensor!")
        print("-" * 50)

    def forward(self, x):
        b, l, s = x.shape
        x = x.reshape(b, l, self.s1, self.s2)
        x = x.reshape(b * l, self.s1, self.s2).unsqueeze(1)

        x = torch.clamp(x, min=-10, max=10)
        x = x.to(next(self.conv_layers.parameters()).device)

        x = self.conv_layers(x)
        x = x.reshape(b, l, 64, self.s1 * self.s2)
        x = x.permute(0, 1, 3, 2)

        return x
"""
Temporal embedding module.

- Encodes time-related features (e.g., hour, weekday) via a linear layer
- Expands each timestamp to match the spatial dimension S
- Output shape: [B, T, S, d_model]
"""

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq):
        super(TimeFeatureEmbedding, self).__init__()
        self.d_model = d_model
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x, S):
        B, T, E = x.shape
        x = self.embed(x)

        emb = []
        for batch in x:
            for time in batch:
                emb.append(torch.stack(S * [time]))
        temporal_emb = torch.stack([temporal for temporal in emb])
        x = temporal_emb.reshape(B, T, S, self.d_model)
        return x

"""
Spatial embedding module.

- Projects 2D location coordinates (e.g., [lat, lon]) to d_model-dimensional space
- Applied per spatial grid location
- Output shape: [B, T, S, d_model]
"""

class SpatialEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SpatialEmbedding, self).__init__()
        self.d_model = d_model
        d_inp = 2
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        x = self.embed(x)
        return x

"""
Main embedding module combining:
1. Value embedding (TokenEmbeddingCONV)
2. Positional embedding (learned)
3. Temporal embedding (TimeFeatureEmbedding)
4. Spatial embedding (SpatialEmbedding)

- Aggregates all embeddings by element-wise addition
- Applies dropout to prevent overfitting
- Output shape: [B, T, S, d_model]
"""

class DataEmbeddingCONV(nn.Module):
    def __init__(self, c_in, d_model, freq='0.5h', dropout=0.1, l=16, k=8):
        super(DataEmbeddingCONV, self).__init__()

        self.value_embedding = TokenEmbeddingCONV(c_in=c_in, d_model=d_model,l = l,k = k)
        self.position_embedding = PositionalEmbeddingCONV(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.spatial_embedding = SpatialEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def debug_tensor(self, tensor, name="Tensor"):
        print(f"Debugging {name}:")
        print(f"Shape: {tensor.shape}")
        if tensor.numel() > 0:  # 确保张量有元素
            print(f"Max: {tensor.max().item()}")
            print(f"Min: {tensor.min().item()}")
            print(f"Mean: {tensor.mean().item()}")
            print(f"Contains NaN: {torch.isnan(tensor).any().item()}")
            print(f"Contains Inf: {torch.isinf(tensor).any().item()}")
        else:
            print("Empty tensor!")
        print("-" * 50)

    def forward(self, x, x_mark, space_mark_x):

        B, T, S, _ = space_mark_x.shape
        x1 = self.value_embedding(x)
        x2 = self.position_embedding(x)

        x3 = self.temporal_embedding(x_mark, S)

        x4 = self.spatial_embedding(space_mark_x)

        x = x1 + x2 + x3 + x4

        return self.dropout(x)


