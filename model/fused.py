import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------ Dilated K Projection ------------------
class DilatedKProjection(nn.Module):
    def __init__(self, dim, dilations=[1, 2, 4]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.fusion = nn.Conv1d(len(dilations) * dim, dim, kernel_size=1)

    def forward(self, x):
        x_t = x.transpose(1, 2)  # [B, C, L]
        outputs = [conv(x_t) for conv in self.convs]
        fused = torch.cat(outputs, dim=1)
        fused = self.fusion(fused)
        return fused.transpose(1, 2)  # [B, L, C]


# ------------------ 完整融合+预测封装类 ------------------
class FusionPredictor(nn.Module):
    """
    将 DPCAttention + 预测线性层 封装在一起
    输入:
        a: Branch1输出 [B,L,C]
        b: Branch2输出 [B,L,C]
    输出:
        fused_features: 融合后的特征 [B,L,2*C]
        logits: 分类logits [B,L]
    """

    def __init__(self, dim, hidden_dim=None, out_dim=1, use_ln=True):
        super().__init__()
        self.dim = dim
        self.use_ln = use_ln
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim

        # DPC Attention Fusion
        self.q_conv3 = nn.Conv1d(dim, dim // 2, kernel_size=3, padding=1, bias=False)
        self.q_conv5 = nn.Conv1d(dim, dim // 2, kernel_size=7, padding=3, bias=False)
        #self.k_proj_base = DilatedKProjection(dim)
        self.k_proj_base = nn.Linear(dim, dim, bias=False)
        self.v_proj_base = nn.Linear(dim, dim, bias=False)
        if use_ln:
            self.ln = nn.LayerNorm(2 * dim)

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(2 * dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )

    def forward(self, a, b, return_features=False):
        # DPC Attention
        q_base = torch.cat([
            self.q_conv3(a.transpose(1, 2)),
            self.q_conv5(a.transpose(1, 2))
        ], dim=1).transpose(1, 2)  # [B,L,C]

        K_base = self.k_proj_base(b)  # [B,L,C]
        V_base = self.v_proj_base(b)  # [B,L,C]

        d_k = q_base.size(-1)
        scores = torch.matmul(q_base, K_base.transpose(1, 2)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_base)  # [B,L,C]

        fused = torch.cat([a, out], dim=-1)  # [B,L,2*C]
        if self.use_ln:
            fused = self.ln(fused)

        # 预测
        logits = self.predictor(fused)  # [B,L,out_dim]
        logits = logits.squeeze(-1)  # [B,L]

        if return_features:
            # 返回融合特征和logits
            return fused, logits
        else:
            # 只返回logits（保持向后兼容）
            return logits


# ------------------ 测试 ------------------
if __name__ == "__main__":
    B, L, C = 4, 300, 512
    a = torch.randn(B, L, C)
    b = torch.randn(B, L, C)
    model = FusionPredictor(dim=C)

    # 测试返回特征和logits
    fused_features, logits = model(a, b, return_features=True)
    print("融合特征形状:", fused_features.shape)  # [B,L,2*C]
    print("分类logits形状:", logits.shape)  # [B,L]

    # 测试只返回logits（向后兼容）
    logits_only = model(a, b, return_features=False)
    print("只返回logits形状:", logits_only.shape)  # [B,L]