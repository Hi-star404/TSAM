import torch
import torch.nn as nn

from models.vlm_models.global_spatial_expert import GlobalSpatialExpert
from models.vlm_models.mmn_cross_routed_spatial_refiner import (
    MMNCrossRoutedSpatialRefiner,
)


class GlobalCompositionalSpatialExpert(nn.Module):
    """Unified global spatial expert (paper-facing single module).

    Stage-V (visual): asymmetric bidirectional attention on verb frames and
    the object vector; light residual only (context is not fed into Stage-C).

    Stage-C (composition): single global relation differentials
        D = (A - I) LN(X),   X' = X + scale * D
    Object side uses a static A_o. Verb/action side uses
        A_v(t) = softmax(S_v + U diag(z_t) V^T)
    where z_t is a low-dimensional temporal structure code (gate summary +
    temporal embedding), projected with a zero-initialized last layer so
    training starts equivalent to a static verb relation matrix.
    """

    def __init__(
            self,
            dim,
            num_object_tokens,
            num_verb_tokens,
            temporal_input_dim=None,
            temporal_gate_dim=4,
            use_visual_stage=True,
            use_composition_stage=True,
            gse_heads=4,
            gse_dropout=0.1,
            gse_alpha_init=0.05,
            gse_beta_init=0.05,
            mmn_bottleneck_dim=128,
            mmn_dropout=0.15,
            mmn_residual_init=0.05,
            mmn_use_local_expert=False,
            mmn_visual_conditioning=False,
            mmn_temporal_conditioning=True,
            mmn_context_residual_mod=False,
            mmn_temporal_relation_rank=8):
        super().__init__()
        self.dim = int(dim)
        self.use_visual_stage = bool(use_visual_stage)
        self.use_composition_stage = bool(use_composition_stage)
        self.mmn_temporal_conditioning = bool(mmn_temporal_conditioning)
        self.temporal_relation_rank = int(mmn_temporal_relation_rank)
        self.temporal_gate_dim = int(temporal_gate_dim)
        temporal_in = int(
            temporal_input_dim if temporal_input_dim is not None else dim
        )
        self.temporal_input_dim = temporal_in
        # Code fed to the verb low-rank calibrator: [emb | gate].
        self.temporal_code_dim = temporal_in + self.temporal_gate_dim
        self._temporal_embedding = None
        self._temporal_gate_weights = None

        self.visual_stage = None
        if self.use_visual_stage:
            self.visual_stage = GlobalSpatialExpert(
                dim=dim,
                num_heads=gse_heads,
                dropout=gse_dropout,
                alpha_init=gse_alpha_init,
                beta_init=gse_beta_init,
            )

        # Legacy attribute: v1 used a feat->emb projector into 900-d MMN
        # context. The new path builds a temporal code instead.
        self.temporal_context_proj = None

        self.composition_stage = None
        if self.use_composition_stage:
            verb_rank = (
                self.temporal_relation_rank
                if self.mmn_temporal_conditioning else 0
            )
            self.composition_stage = MMNCrossRoutedSpatialRefiner(
                num_object_tokens=num_object_tokens,
                num_verb_tokens=num_verb_tokens,
                dim=dim,
                bottleneck_dim=mmn_bottleneck_dim,
                dropout=mmn_dropout,
                residual_init=mmn_residual_init,
                use_local_expert=False,
                visual_conditioning=False,
                use_context_residual_mod=False,
                temporal_relation_rank=verb_rank,
                temporal_code_dim=(
                    self.temporal_code_dim if verb_rank > 0 else None
                ),
            )

    def clear_context(self):
        self._temporal_embedding = None
        self._temporal_gate_weights = None

    def set_temporal_context(
            self, temporal_embedding, temporal_gate_weights=None):
        """Cache temporal video embedding (+ optional gate) for verb A_v(t)."""
        self._temporal_embedding = temporal_embedding
        self._temporal_gate_weights = temporal_gate_weights

    def _build_temporal_code(self, batch_size, ref):
        if not self.mmn_temporal_conditioning:
            return None
        if self._temporal_embedding is None:
            emb = torch.zeros(
                batch_size,
                self.temporal_input_dim,
                device=ref.device,
                dtype=ref.dtype,
            )
        else:
            if self._temporal_embedding.shape[0] != batch_size:
                raise ValueError(
                    "temporal_embedding batch does not match composition tokens."
                )
            emb = self._temporal_embedding.to(dtype=ref.dtype)
            if emb.ndim != 2 or emb.shape[-1] != self.temporal_input_dim:
                raise ValueError(
                    f"Expected temporal_embedding [B, {self.temporal_input_dim}], "
                    f"got {tuple(emb.shape)}."
                )

        if self._temporal_gate_weights is None:
            gate = torch.zeros(
                batch_size,
                self.temporal_gate_dim,
                device=ref.device,
                dtype=ref.dtype,
            )
        else:
            gate = self._temporal_gate_weights.to(dtype=ref.dtype)
            if gate.shape[0] != batch_size:
                raise ValueError(
                    "temporal_gate_weights batch does not match composition tokens."
                )
            if gate.ndim != 2:
                raise ValueError(
                    f"Expected temporal_gate_weights [B, K], got {tuple(gate.shape)}."
                )
            if gate.shape[-1] < self.temporal_gate_dim:
                pad = torch.zeros(
                    batch_size,
                    self.temporal_gate_dim - gate.shape[-1],
                    device=gate.device,
                    dtype=gate.dtype,
                )
                gate = torch.cat([gate, pad], dim=-1)
            elif gate.shape[-1] > self.temporal_gate_dim:
                gate = gate[:, : self.temporal_gate_dim]
        return torch.cat([emb, gate], dim=-1)

    def refine_visual(self, v_feat_t, o_feat):
        """Run Stage-V. Context is not forwarded into Stage-C."""
        if self.visual_stage is None:
            return v_feat_t, o_feat
        v_out, o_out, _context = self.visual_stage(
            v_feat_t, o_feat, return_context=True
        )
        return v_out, o_out

    def refine_composition(self, object_conditioned_tokens, verb_conditioned_tokens):
        """Run Stage-C; temporal code calibrates verb relations only."""
        if self.composition_stage is None:
            return object_conditioned_tokens, verb_conditioned_tokens
        temporal_code = self._build_temporal_code(
            object_conditioned_tokens.shape[0],
            object_conditioned_tokens,
        )
        return self.composition_stage(
            object_conditioned_tokens,
            verb_conditioned_tokens,
            visual_context=None,
            temporal_code=temporal_code,
        )
