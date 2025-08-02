
import torch
import torch.nn as nn
from models.optionT import compute_optimal_transport
from models.encoder import Encoder, EncoderLayer
from models.attn import AttentionLayer, ConditionedAttention
from models.embed import DataEmbeddingCONV

"""
Channel-guided attention module for feature weighting across channels.
Used as a dynamic fusion gate in the local-global integration step.
"""

class ChannelGuidedAttention(nn.Module):

    def __init__(self, in_channels):
        super(ChannelGuidedAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

"""
P-GOT model:
Physics-guided Optimal Transport framework for spatiotemporal traffic prediction.
Integrates:
- Conv2D-based local spatial-temporal embedding
- Optimal transport guided attention
- Global MLP module
- Gated local-global fusion
- Final regression head
"""

class Pgot(nn.Module):
    def __init__(self, enc_in, c_out, seq_len, label_len, out_len, d_model=64, n_heads=8, e_layers=3, d_ff=512,l = 16, k=8,
                 dropout=0.2, freq='h', activation='gelu',
                 output_attention=False, device=torch.device('cuda:0')):
        """
        Model initialization:
        - Builds embedding layer, encoder stack, spatial/temporal MLPs
        - Defines 1x1 conv, projection layers, gated fusion (CGA), and regressor
        """

        super(Pgot, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention
        self.label_len = label_len
        self.l = l
        self.k = k
        self.Conv2D = DataEmbeddingCONV(enc_in, d_model,freq, dropout, l, k)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ConditionedAttention(attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ]
        )
        self.Spatiomlp = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.Temporalmlp = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.projection = nn.Linear(d_model, d_model)
        self.sequence_projection = nn.Linear(l* k * seq_len, out_len * l*k)
        self.conv1x1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.output_regressor = nn.Linear(d_model, c_out)
        self.cga = ChannelGuidedAttention(d_model)  # CGA 模块
        self.conv1x1 = nn.Conv2d(d_model, d_model, kernel_size=1)  # 1x1 卷积
        self.relu = nn.ReLU()

    """
    Creates 2D grid-based spatial adjacency matrix (8-neighborhood).
    Used for physics-guided OT computation between spatial regions.
    """

    def build_grid_adjacency(self,height=16, width=8):
        num_nodes = height * width
        A = torch.zeros((num_nodes, num_nodes))
        for i in range(height):
            for j in range(width):
                index = i * width + j
                directions = [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1)
                ]
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_index = ni * width + nj
                        A[index, neighbor_index] = 1
        return A

    """
    Model forward pass:

    1. Local dynamics extraction:
       - Generate spatial-temporal embeddings
       - Compute optimal transport matrix from potential evolution
       - Apply OT-guided attention encoding

    2. Global representation extraction:
       - Apply spatial and temporal MLPs independently
       - Generate a global latent representation

    3. Local-global feature fusion:
       - Fuse local and global features using channel-guided attention (CGA)
       - Refine with 1x1 convolution

    4. Prediction:
       - Linear projection and regression to output future traffic flow
    """

    def forward(self, x_enc, x_mark_enc, space_mark_x):
        # Local dynamic flow pattern extraction
        H_embedding = self.Conv2D(x_enc, x_mark_enc, space_mark_x)  # Spatiotemporal features
        B, T, S, D = H_embedding.shape
        A = self.build_grid_adjacency(height=self.l, width=self.k)
        T_guided = compute_optimal_transport(H_embedding, A) # Optimal transport matrix
        H_local, _ = self.encoder(H_embedding, T_guided)  # Output from the local encoder module
        H_local = H_local.permute(0, 1, 3, 2).reshape(B * T, D, self.l, self.k)

        # Global spatio-temporal MLP feature extraction
        H_spatial_in = H_embedding.reshape(B * T, S, D)
        H_spatial_out = self.Spatiomlp(H_spatial_in).reshape(B, T, S, D)

        H_temporal_in = H_spatial_out.permute(0, 2, 1, 3).reshape(B * S, T, D)
        H_temporal_out = self.Temporalmlp(H_temporal_in).reshape(B, S, T, D).permute(0, 2, 1, 3)

        H_global_proj = self.projection(H_temporal_out.reshape(B, T * S, D)).reshape(B * T, D, self.l, self.k)

        # Feature fusion (gated weighting)
        H_fusion_input = H_local + H_global_proj
        W_gate = self.cga(H_fusion_input)
        H_fused = W_gate * H_local + (1 - W_gate) * H_global_proj
        H_refined = self.conv1x1(H_fused)

        # Prediction
        BT, D, H, W = H_refined.shape
        Y_feat = H_refined.reshape(B, T * S, D).permute(0, 2, 1)
        Y_feat = self.sequence_projection(Y_feat)
        Y_feat = Y_feat.reshape(B, D, self.pred_len, S).permute(0, 2, 3, 1)
        Y_feat = self.relu(Y_feat)

        Y_feat = Y_feat.reshape(B, self.pred_len * S, D)
        Y_out = self.output_regressor(Y_feat)
        Y_out = Y_out.reshape(B, self.pred_len, S, 1).squeeze(dim=-1)

        return Y_out
