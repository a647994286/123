import torch
import torch.nn as nn
class ConditionedAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, output_attention=False):
        super(ConditionedAttention, self).__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, OT_matrices=None):
        """
        Applies attention between each time step t and its next time step (t+1), modulated by partial optimal transport (OT).

        Args:
            queries: Query tensor of shape (B, T, S, H, D)
            keys: Key tensor of shape (B, T, S, H, D)
            values: Value tensor of shape (B, T, S, H, D)
            attn_mask: Optional attention mask (B, T, S, S)
            OT_matrices: Optional OT bias tensor (B, T, S, S)

        Returns:
            output: Attention output of shape (B, T, S, H, D)
            all_attn (optional): Attention scores for each timestep
        """
        B, T, S, H, D = queries.shape
        scale = 1. / torch.sqrt(torch.tensor(D, dtype=torch.float32, device=queries.device))

        output = torch.zeros_like(queries)
        all_attn = []

        for t in range(T):
            next_t = (t + 1) % T

            Q_t = queries[:, t, :, :, :]   # (B, S, H, D)
            K_t1 = keys[:, next_t, :, :, :]
            V_t1 = values[:, next_t, :, :, :]


            Q_t = Q_t.permute(0, 2, 1, 3)    # (B, H, S, D)
            K_t1 = K_t1.permute(0, 2, 3, 1)  # (B, H, D, S)
            V_t1 = V_t1.permute(0, 2, 1, 3)  # (B, H, S, D)

            scores = torch.matmul(Q_t, K_t1) * scale  # (B, H, S, S)

            if OT_matrices is not None:
                OT_t = OT_matrices[:, t, :, :].unsqueeze(1)  # (B, 1, S, S)
                OT_bias = torch.log(OT_t + 1e-8)
                scores = scores + OT_bias

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            context = torch.matmul(attn, V_t1)  # (B, H, S, D)
            output[:, t, :, :, :] = context.permute(0, 2, 1, 3)

            if self.output_attention:
                all_attn.append(attn)

        if self.output_attention:
            return output, torch.stack(all_attn, dim=1)
        else:
            return output, None




class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None, OT_matrices=None):
        """
        Applies multi-head attention with optional OT guidance.

        Args:
            queries: (B, T, S, D) - input queries
            keys: (B, T, S, D) - input keys
            values: (B, T, S, D) - input values
            attn_mask: Optional attention mask (B, T, S, S)
            OT_matrices: Optional optimal transport matrices (B, T, S, S)

        Returns:
            output: Final projected attention output (B, T, S, D)
            attn: Optional attention scores
        """
        B, T, S, D = queries.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, T, S, H, -1)
        keys = self.key_projection(keys).view(B, T, S, H, -1)
        values = self.value_projection(values).view(B, T, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            OT_matrices=OT_matrices
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()

        out = out.view(B, T, S, -1)
        return self.out_projection(out), attn

