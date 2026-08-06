import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialGlobalRelationExpert(nn.Module):
    """Single global relation differential: X' = X + scale * ((A-I) LN(X)).

    No local MLP, no identity hop, no multi-expert routing. Optional low-rank
    temporal calibration of the relation logits:
        A(t) = softmax(S + U diag(z_t) V^T)
    With z_t=0 the dynamic term vanishes and A collapses to softmax(S).
    """

    def __init__(
            self,
            num_tokens,
            dim,
            residual_init=0.05,
            dropout=0.15,
            temporal_rank=0,
            temporal_code_dim=None):
        super().__init__()
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive.")
        self.num_tokens = int(num_tokens)
        self.dim = int(dim)
        self.temporal_rank = int(temporal_rank)
        self.norm = nn.LayerNorm(dim)
        self.relation_logits = nn.Parameter(
            torch.empty(self.num_tokens, self.num_tokens)
        )
        nn.init.normal_(self.relation_logits, std=0.02)
        self.output_dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )

        self.relation_U = None
        self.relation_V = None
        self.temporal_code_proj = None
        if self.temporal_rank > 0:
            if temporal_code_dim is None:
                raise ValueError(
                    "temporal_code_dim is required when temporal_rank > 0."
                )
            self.relation_U = nn.Parameter(
                torch.empty(self.num_tokens, self.temporal_rank)
            )
            self.relation_V = nn.Parameter(
                torch.empty(self.num_tokens, self.temporal_rank)
            )
            nn.init.normal_(self.relation_U, std=0.02)
            nn.init.normal_(self.relation_V, std=0.02)
            # Maps an external temporal code to z_t; last layer zero-init so
            # A(t) starts exactly as the static softmax(S).
            hidden = max(int(temporal_code_dim), self.temporal_rank * 2)
            self.temporal_code_proj = nn.Sequential(
                nn.LayerNorm(int(temporal_code_dim)),
                nn.Linear(int(temporal_code_dim), hidden),
                nn.GELU(),
                nn.Linear(hidden, self.temporal_rank),
            )
            nn.init.zeros_(self.temporal_code_proj[-1].weight)
            nn.init.zeros_(self.temporal_code_proj[-1].bias)

    def _relation_matrix(self, batch_size, ref, temporal_code=None):
        logits = self.relation_logits
        if (
                self.temporal_rank > 0
                and self.relation_U is not None
                and self.relation_V is not None
                and self.temporal_code_proj is not None
        ):
            if temporal_code is None:
                z_t = torch.zeros(
                    batch_size,
                    self.temporal_rank,
                    device=ref.device,
                    dtype=ref.dtype,
                )
            else:
                if temporal_code.shape[0] != batch_size:
                    raise ValueError(
                        "temporal_code batch does not match composition tokens."
                    )
                z_t = self.temporal_code_proj(
                    temporal_code.float()
                ).to(dtype=ref.dtype)
            # [B, N, N] low-rank correction: U diag(z) V^T
            # einsum: bnr,br,mr -> bnm  with V as [N,R]
            dynamic = torch.einsum(
                "nr,br,mr->bnm",
                self.relation_U.to(dtype=ref.dtype),
                z_t,
                self.relation_V.to(dtype=ref.dtype),
            )
            logits = logits.to(dtype=ref.dtype).unsqueeze(0) + dynamic
            return F.softmax(logits, dim=-1)
        static = F.softmax(logits.to(dtype=ref.dtype), dim=-1)
        return static.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, tokens, temporal_code=None):
        if tokens.ndim != 3 or tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected [B, {self.num_tokens}, C] tokens, got {tokens.shape}."
            )
        x = self.norm(tokens)
        relation = self._relation_matrix(
            tokens.shape[0], tokens, temporal_code=temporal_code
        )
        # relation: [B, N, N], x: [B, N, C] -> [B, N, C]
        related = torch.einsum("bnm,bmc->bnc", relation, x)
        delta = related - x
        scale = torch.tanh(self.residual_scale)
        return tokens + scale * self.output_dropout(delta)


class MMNCrossRoutedSpatialRefiner(nn.Module):
    """Stage-C: temporally calibrated single-relation differential experts.

    Object branch: static global relation differential only.
    Verb/action branch: optional low-rank temporal calibration of A_v(t).
    Residual scales are independent of temporal codes (no dual control).
    """

    def __init__(
            self,
            num_object_tokens,
            num_verb_tokens,
            dim,
            bottleneck_dim=128,
            dropout=0.15,
            residual_init=0.05,
            use_local_expert=False,
            visual_conditioning=False,
            condition_dim=None,
            use_context_residual_mod=False,
            temporal_relation_rank=8,
            temporal_code_dim=None):
        super().__init__()
        del bottleneck_dim, use_local_expert, condition_dim
        del use_context_residual_mod
        self.dim = int(dim)
        # Kept for logging / ablation scripts; Stage-C no longer uses
        # 900-d visual/temporal context routing.
        self.visual_conditioning = bool(visual_conditioning)
        self.context_residual_mod = None
        self.temporal_relation_rank = int(temporal_relation_rank)

        self.object_expert = DifferentialGlobalRelationExpert(
            num_tokens=num_object_tokens,
            dim=dim,
            residual_init=residual_init,
            dropout=dropout,
            temporal_rank=0,
        )
        self.verb_expert = DifferentialGlobalRelationExpert(
            num_tokens=num_verb_tokens,
            dim=dim,
            residual_init=residual_init,
            dropout=dropout,
            temporal_rank=self.temporal_relation_rank,
            temporal_code_dim=temporal_code_dim,
        )

        # Compatibility aliases used by older analysis / logging code.
        self.object_conditioned_bank = self.object_expert
        self.verb_conditioned_bank = self.verb_expert

    def forward(
            self,
            object_conditioned_tokens,
            verb_conditioned_tokens,
            visual_context=None,
            temporal_code=None):
        del visual_context
        object_output = self.object_expert(object_conditioned_tokens)
        verb_output = self.verb_expert(
            verb_conditioned_tokens, temporal_code=temporal_code
        )
        return object_output, verb_output
