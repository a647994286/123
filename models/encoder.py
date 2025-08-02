import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, OT_matrices, attn_mask=None):
        """
        Args:
            x: Input tensor of shape (B, T, S, D), where
               B = batch size, T = time steps, S = spatial nodes, D = feature dimension
            OT_matrices: Optimal Transport matrices of shape (B, T, S, S)
            attn_mask: Optional attention mask (B, T, S, S)
        """
        B, T, S, D = x.shape

        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, OT_matrices=OT_matrices)

        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = y.reshape(B, T * S, D)  # (B, T×S, D)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # (B, D, T×S)
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # (B, T×S, D)
        y = y.reshape(B, T, S, D)
        y = self.norm2(x + y)

        return y, attn



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, OT_matrices, attn_mask=None):
        """
               Args:
                   x: Input spatiotemporal features of shape (B, T, S, D)
                   OT_matrices: Optimal Transport matrices between adjacent time steps (B, T, S, S)
                   attn_mask: Optional attention mask (B, T, S, S)
               """
        attns = []

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, OT_matrices=OT_matrices, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
