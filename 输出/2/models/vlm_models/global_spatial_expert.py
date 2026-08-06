import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSpatialExpert(nn.Module):
    """Stage-V of the unified global spatial expert.

    Asymmetric bidirectional attention over verb frames and the object
    vector. The visual residual is intentionally small so that most of the
    accuracy gain is deferred to the composition stage (MMN), which is
    conditioned on the attention deltas returned here.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1, alpha_init=0.02, beta_init=0.02):
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

    def forward(self, v_feat_t, o_feat, return_context=True):
        v = v_feat_t.transpose(1, 2).contiguous()
        o = o_feat.unsqueeze(1)

        v_n = self.norm_v(v)
        o_n = self.norm_o(o)

        qv = self._split_heads(self.q_v(v_n))
        ko = self._split_heads(self.k_o(o_n))
        vo = self._split_heads(self.v_o(o_n))

        gate_vo = torch.sigmoid((qv @ ko.transpose(-2, -1)) * self.scale)
        delta_v = self._merge_heads(gate_vo * vo)
        delta_v = self.drop(self.proj_v(delta_v))

        qo = self._split_heads(self.q_o(o_n))
        kv = self._split_heads(self.k_v(v_n))
        vv = self._split_heads(self.v_v(v_n))

        attn_ov = (qo @ kv.transpose(-2, -1)) * self.scale
        attn_ov = self.drop(F.softmax(attn_ov, dim=-1))
        delta_o = self._merge_heads(attn_ov @ vv)
        delta_o = self.proj_o(delta_o)

        # Light visual residual: Stage-V proposes relation context; Stage-C
        # (composition refiner) applies the main calibrated residual.
        v = v + self.alpha * delta_v
        o = o + self.beta * delta_o

        v = v + self.ffn_v(v)
        o = o + self.ffn_o(o)

        v_out = v.transpose(1, 2).contiguous()
        o_out = o.squeeze(1)
        if not return_context:
            return v_out, o_out

        # Context consumed by the composition stage: pooled verb update and
        # object update from the cross-branch attention (before FFN).
        context = torch.cat([delta_v.mean(dim=1), delta_o.squeeze(1)], dim=-1)
        return v_out, o_out, context
