import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSpatialExpert(nn.Module):
    """Non-convolutional global spatial expert for cross-branch relation enhancement."""

    def __init__(self, dim, num_heads=4, dropout=0.1, alpha_init=0.05, beta_init=0.05):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_v = nn.LayerNorm(dim)
        self.norm_o = nn.LayerNorm(dim)

        self.q_v = nn.Linear(dim, dim, bias=False)
        self.k_o = nn.Linear(dim, dim, bias=False)
        self.v_o = nn.Linear(dim, dim, bias=False)

        self.q_o = nn.Linear(dim, dim, bias=False)
        self.k_v = nn.Linear(dim, dim, bias=False)
        self.v_v = nn.Linear(dim, dim, bias=False)

        self.proj_v = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)

        self.drop = nn.Dropout(dropout)

        self.ffn_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.ffn_o = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def _split_heads(self, x):
        b, n, c = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def forward(self, v_feat_t, o_feat):
        # v_feat_t: [B, C, T], o_feat: [B, C]
        v = v_feat_t.transpose(1, 2).contiguous()  # [B, T, C]
        o = o_feat.unsqueeze(1)  # [B, 1, C]

        v_n = self.norm_v(v)
        o_n = self.norm_o(o)

        # Dynamic tokens query one static object token. Since there is only one key,
        # softmax would collapse to 1.0; use independent sigmoid gates instead.
        qv = self._split_heads(self.q_v(v_n))
        ko = self._split_heads(self.k_o(o_n))
        vo = self._split_heads(self.v_o(o_n))

        gate_vo = torch.sigmoid((qv @ ko.transpose(-2, -1)) * self.scale)
        delta_v = self._merge_heads(gate_vo * vo)
        delta_v = self.drop(self.proj_v(delta_v))

        # Static object token attends over the full dynamic timeline. Keeping all
        # temporal keys lets softmax select highlight frames instead of collapsing.
        qo = self._split_heads(self.q_o(o_n))
        kv = self._split_heads(self.k_v(v_n))
        vv = self._split_heads(self.v_v(v_n))

        attn_ov = (qo @ kv.transpose(-2, -1)) * self.scale
        attn_ov = self.drop(F.softmax(attn_ov, dim=-1))
        delta_o = self._merge_heads(attn_ov @ vv)
        delta_o = self.proj_o(delta_o)

        v = v + self.alpha * delta_v
        o = o + self.beta * delta_o

        v = v + self.ffn_v(v)
        o = o + self.ffn_o(o)

        return v.transpose(1, 2).contiguous(), o.squeeze(1)