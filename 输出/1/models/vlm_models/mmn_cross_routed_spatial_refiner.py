import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class MMNRelationExpertBank(nn.Module):
    """MMN-inspired identity-local and learned-global relation experts."""

    def __init__(self, num_tokens, dim, bottleneck_dim=128, dropout=0.15):
        super().__init__()
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive.")

        self.num_tokens = int(num_tokens)
        self.norm = nn.LayerNorm(dim)
        self.local_expert = BottleneckAdapter(dim, bottleneck_dim, dropout)
        self.global_relation_logits = nn.Parameter(
            torch.empty(self.num_tokens, self.num_tokens)
        )
        nn.init.normal_(self.global_relation_logits, std=0.02)
        self.global_expert = BottleneckAdapter(dim, bottleneck_dim, dropout)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected [B, {self.num_tokens}, C] tokens, got {tokens.shape}."
            )

        x = self.norm(tokens)
        local_output = self.local_expert(x)

        global_relation = F.softmax(self.global_relation_logits, dim=-1)
        global_features = torch.einsum("nm,bmc->bnc", global_relation, x)
        global_output = self.global_expert(global_features)

        return torch.stack([local_output, global_output], dim=1)


class MMNCrossRoutedSpatialRefiner(nn.Module):
    """Opposite-branch routing over local/global relation experts with residual protection."""

    def __init__(
            self,
            num_object_tokens,
            num_verb_tokens,
            dim,
            bottleneck_dim=128,
            dropout=0.15,
            residual_init=0.05):
        super().__init__()
        common = dict(
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.object_conditioned_bank = MMNRelationExpertBank(
            num_tokens=num_object_tokens,
            **common,
        )
        self.verb_conditioned_bank = MMNRelationExpertBank(
            num_tokens=num_verb_tokens,
            **common,
        )

        route_hidden = max(dim // 2, 64)
        self.verb_to_object_route = self._make_route(dim, route_hidden, dropout)
        self.object_to_verb_route = self._make_route(dim, route_hidden, dropout)
        self.object_update_norm = nn.LayerNorm(dim)
        self.verb_update_norm = nn.LayerNorm(dim)
        self.object_projection = nn.Linear(dim, dim, bias=False)
        self.verb_projection = nn.Linear(dim, dim, bias=False)
        self.output_dropout = nn.Dropout(dropout)
        self.object_residual_scale = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )
        self.verb_residual_scale = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )

    @staticmethod
    def _make_route(dim, hidden_dim, dropout):
        route = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(route[-1].weight)
        nn.init.zeros_(route[-1].bias)
        return route

    @staticmethod
    def _expert_summary(expert_outputs):
        batch_size = expert_outputs.shape[0]
        return expert_outputs.mean(dim=2).reshape(batch_size, -1)

    def forward(self, object_conditioned_tokens, verb_conditioned_tokens):
        object_experts = self.object_conditioned_bank(object_conditioned_tokens)
        verb_experts = self.verb_conditioned_bank(verb_conditioned_tokens)

        object_weights = F.softmax(
            self.verb_to_object_route(self._expert_summary(verb_experts)),
            dim=-1,
        )
        verb_weights = F.softmax(
            self.object_to_verb_route(self._expert_summary(object_experts)),
            dim=-1,
        )

        object_update = torch.einsum(
            "bk,bknc->bnc", object_weights, object_experts
        )
        verb_update = torch.einsum(
            "bk,bknc->bnc", verb_weights, verb_experts
        )
        object_update = self.object_projection(self.object_update_norm(object_update))
        verb_update = self.verb_projection(self.verb_update_norm(verb_update))

        object_output = (
            object_conditioned_tokens
            + torch.tanh(self.object_residual_scale)
            * self.output_dropout(object_update)
        )
        verb_output = (
            verb_conditioned_tokens
            + torch.tanh(self.verb_residual_scale)
            * self.output_dropout(verb_update)
        )
        return object_output, verb_output
