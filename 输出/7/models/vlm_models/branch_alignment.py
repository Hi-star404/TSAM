import math

import torch
import torch.nn as nn


def scale_gradient(x, scale):
    """Keep the forward value unchanged while scaling its backward gradient."""
    scale = float(scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("gradient scale must be in [0, 1].")
    return x.detach() + scale * (x - x.detach())


class LowRankResidualAligner(nn.Module):
    """Identity-initialized channel alignment with a bounded residual."""

    def __init__(
            self,
            dim,
            bottleneck_dim=64,
            dropout=0.1,
            residual_init=0.05,
            residual_max=0.2):
        super().__init__()
        if bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be positive.")
        if residual_max <= 0.0:
            raise ValueError("residual_max must be positive.")
        if abs(residual_init) >= residual_max:
            raise ValueError("abs(residual_init) must be below residual_max.")

        self.residual_max = float(residual_max)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        initial_ratio = float(residual_init) / self.residual_max
        self.residual_logit = nn.Parameter(
            torch.tensor(math.atanh(initial_ratio), dtype=torch.float32)
        )

    def residual_scale(self):
        return self.residual_max * torch.tanh(self.residual_logit)

    def forward(self, x):
        delta = self.up(
            self.dropout(self.activation(self.down(self.norm(x))))
        )
        return x + self.residual_scale() * delta


class GradientDecoupledBranchAlignment(nn.Module):
    """Align temporal VE2/OE2 features before GSE without overwriting them."""

    def __init__(
            self,
            dim,
            bottleneck_dim=64,
            dropout=0.1,
            residual_init=0.05,
            residual_max=0.2,
            gradient_scale=0.25):
        super().__init__()
        if not 0.0 <= float(gradient_scale) <= 1.0:
            raise ValueError("gradient_scale must be in [0, 1].")
        self.gradient_scale = float(gradient_scale)
        common = dict(
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            residual_init=residual_init,
            residual_max=residual_max,
        )
        self.verb_aligner = LowRankResidualAligner(**common)
        self.object_aligner = LowRankResidualAligner(**common)

    def forward(self, verb_features_t, object_features):
        # The aligners learn from the global branch, while VE2/OE2 receive only
        # the configured fraction of that branch's gradient.
        verb_features = scale_gradient(
            verb_features_t.transpose(1, 2).contiguous(),
            self.gradient_scale,
        )
        object_features = scale_gradient(
            object_features,
            self.gradient_scale,
        )
        verb_aligned = self.verb_aligner(verb_features)
        object_aligned = self.object_aligner(object_features)
        return (
            verb_aligned.transpose(1, 2).contiguous(),
            object_aligned,
        )

    def current_scales(self):
        return {
            "verb": float(self.verb_aligner.residual_scale().detach().cpu()),
            "object": float(
                self.object_aligner.residual_scale().detach().cpu()
            ),
            "gradient": self.gradient_scale,
        }
