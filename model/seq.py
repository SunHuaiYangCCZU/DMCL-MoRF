import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------Transformer---------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, L, C = q.shape
        q = self.q_proj(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, L, C)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        attn_out = self.attn(q, k, v)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])

    def forward(self, q, k, v):
        x = q
        for layer in self.layers:
            x = layer(x, k, v)
        return x


# ---------------------------特征融合模块---------------------------------
#
class FeatureFusion3D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 关系建模卷积（核心创新点）
        self.rel_conv = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.act = nn.Sigmoid()

    def forward(self, x1, y1):
        """
        x1, y1: [B, C, L]
        """

        # 1. 构造关系特征
        rel = torch.abs(x1 - y1)          # |E − T| → [B, C, L]

        # 2. 关系驱动门控
        gate = self.act(self.rel_conv(rel))  # [B, C, L]

        # 3. 对两个特征进行一致调制
        x1_g = x1 * gate
        y1_g = y1 * gate

        # 4. 融合
        fused = x1_g + y1_g

        return x1_g, y1_g, fused


class DimAdjust3D(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)  # [B, out_channels, L]


# ---------------------------第一分支网络---------------------------------
class Branch1(nn.Module):
    def __init__(self, x_dim=1280, y_dim=1024, hidden_dim=512, seq_len=300, transformer_depth=3, num_heads=8):
        super().__init__()
        self.x_adjust = DimAdjust3D(x_dim, hidden_dim)
        self.y_adjust = DimAdjust3D(y_dim, hidden_dim)
        self.fusion = FeatureFusion3D(hidden_dim)
        self.transformer = Transformer(hidden_dim, transformer_depth, num_heads)

    def forward(self, x, y, return_intermediate=False):
        # x, y: [B,L,C]
        x1 = self.x_adjust(x.transpose(1, 2))  # [B, C, L]
        y1 = self.y_adjust(y.transpose(1, 2))  # [B, C, L]

        # 特征融合 + 注意力
        x_attn, y_attn, fused = self.fusion(x1, y1)  # [B,C,L] x3
        # 转置到 [B,L,C]
        x_attn_t = x_attn.transpose(1, 2)
        y_attn_t = y_attn.transpose(1, 2)
        fused_t = fused.transpose(1, 2)

        # Transformer以三路特征作为 Q,K,V
        # out = self.transformer(x_attn_t, y_attn_t, fused_t)  # [B,L,C]
        out = self.transformer(fused_t, fused_t, fused_t)  # [B,L,C]

        if return_intermediate:
            # 返回四个特征：
            # 1. 最终输出 (out)
            # 2. 特征融合结果 (fused_t)
            # 3. 中间特征1 - 经过注意力加权的x特征 (x_attn_t)
            # 4. 中间特征2 - 经过注意力加权的y特征 (y_attn_t)
            return out, fused_t, x_attn_t, y_attn_t
        else:
            # 只返回 Transformer 输出
            return out


# ---------------------------测试---------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Branch1().to(device)
    X = torch.randn(4, 300, 1280).to(device)
    Y = torch.randn(4, 300, 1024).to(device)

    # 测试返回中间特征
    final_out, fused_features, intermediate_x, intermediate_y = model(X, Y, return_intermediate=True)
    print("Final output:", final_out.shape)
    print("Fused features:", fused_features.shape)
    print("Intermediate X features:", intermediate_x.shape)
    print("Intermediate Y features:", intermediate_y.shape)

    # 测试不返回中间特征（向后兼容）
    final_out_only = model(X, Y, return_intermediate=False)
    print("Final output only:", final_out_only.shape)