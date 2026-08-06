import math
import re

import torch
import torch.nn as nn

from clip import clip

from models.vlm_models.text_learner import get_text_learner
from models.vlm_models.global_spatial_expert import GlobalSpatialExpert
from models.vlm_models.mmn_cross_routed_spatial_refiner import MMNCrossRoutedSpatialRefiner

import torch.nn.functional as F

from einops import rearrange


_SOMETHING_PATTERN = re.compile(
    r"\[\s*something(?:\s+else)?\s*\]",
    flags=re.IGNORECASE,
)


def build_role_robust_natural_prompts(verb, obj):
    """Create joint prompts without assuming every placeholder has one role.

    The third prompt is deliberately verb-centric so the natural text bank
    carries stronger action evidence for the weak verb component.
    """
    action = str(verb).strip().rstrip(".").lower()
    object_name = str(obj).strip().lower()
    slot_count = len(_SOMETHING_PATTERN.findall(action))
    verb_focus_action = _SOMETHING_PATTERN.sub("something", action)
    verb_focus_prompt = (
        "focusing on the action, a person is {}".format(verb_focus_action)
    )

    if slot_count == 1:
        filled_action = _SOMETHING_PATTERN.sub(
            object_name,
            action,
            count=1,
        )
        return (
            "a video of a person {}".format(filled_action),
            "the action is a person {}".format(filled_action),
            verb_focus_prompt,
        )

    if slot_count > 1:
        replacement_index = [0]

        def replace_slot(_match):
            replacement_index[0] += 1
            if replacement_index[0] == 1:
                return object_name
            return "another object"

        first_role_action = _SOMETHING_PATTERN.sub(
            replace_slot,
            action,
        )
        neutral_action = _SOMETHING_PATTERN.sub(
            "an object",
            action,
        )
        return (
            "a video of a person {}".format(first_role_action),
            (
                "a video involving {}; a person is {}"
                .format(object_name, neutral_action)
            ),
            verb_focus_prompt,
        )

    return (
        "a video of {}; a person is {}".format(object_name, action),
        "a video involving the object {} and the action {}".format(
            object_name,
            action,
        ),
        "focusing on the action {}, involving {}".format(
            action,
            object_name,
        ),
    )


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class TemporalExpertConv1d(nn.Module):
    """Prior-bounded mixture of temporal convolution experts."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes=(3, 5, 7, 9),
            bias=True,
            temperature=1.0,
            use_gate_norm=False,
            gate_prior=None,
            fixed_gate_weights=None,
            gate_residual_scale=None):
        super().__init__()
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if not self.kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one value.")
        if any(k < 1 or k % 2 == 0 for k in self.kernel_sizes):
            raise ValueError("Temporal expert kernel sizes must be positive odd numbers.")

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm1d(in_channels),
                nn.GELU(),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
            )
            for k in self.kernel_sizes
        ])

        gate_in_dim = 2 * in_channels
        gate_hidden = max(in_channels, 16)
        self.gate_norm = (
            nn.LayerNorm(gate_in_dim) if use_gate_norm else nn.Identity()
        )
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, len(self.kernel_sizes)),
        )
        self.gate_temperature = max(float(temperature), 1e-6)
        self._initialize_gate_prior(gate_prior)
        self._initialize_fixed_gate_weights(fixed_gate_weights)
        self.gate_residual_scale = (
            None
            if gate_residual_scale is None
            else float(gate_residual_scale)
        )
        if (
            self.gate_residual_scale is not None
            and self.gate_residual_scale < 0.0
        ):
            raise ValueError("gate_residual_scale must be non-negative.")
        if (
            self.gate_residual_scale is not None
            and self.gate_prior is None
        ):
            raise ValueError(
                "gate_prior is required when gate_residual_scale is enabled."
            )
        if (
            self.gate_residual_scale is not None
            and self.fixed_gate_weights is not None
        ):
            raise ValueError(
                "fixed_gate_weights and gate_residual_scale are mutually exclusive."
            )

        if self.gate_residual_scale is not None:
            output_layer = self.gate[-1]
            with torch.no_grad():
                output_layer.weight.zero_()
                output_layer.bias.zero_()

        if self.fixed_gate_weights is not None:
            for parameter in self.gate.parameters():
                parameter.requires_grad_(False)

    def _initialize_gate_prior(self, gate_prior):
        self.register_buffer("gate_prior", None)
        if gate_prior is None:
            return
        prior = torch.as_tensor(gate_prior, dtype=torch.float32)
        if prior.numel() != len(self.kernel_sizes):
            raise ValueError(
                "gate_prior must have one value for each temporal expert."
            )
        if not torch.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError("gate_prior values must be finite and positive.")
        prior = prior / prior.sum()
        output_layer = self.gate[-1]
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.copy_(
                self.gate_temperature * prior.log()
            )
        self.gate_prior = prior

    def _initialize_fixed_gate_weights(self, fixed_gate_weights):
        self.register_buffer("fixed_gate_weights", None)
        if fixed_gate_weights is None:
            return
        weights = torch.as_tensor(fixed_gate_weights, dtype=torch.float32)
        if weights.numel() != len(self.kernel_sizes):
            raise ValueError(
                "fixed_gate_weights must have one value for each temporal expert."
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                "fixed_gate_weights values must be finite and non-negative."
            )
        if weights.sum() <= 0:
            raise ValueError("fixed_gate_weights must have a positive sum.")
        self.fixed_gate_weights = weights / weights.sum()

    def forward(self, x, return_gate=False):
        if self.fixed_gate_weights is not None:
            gate_weights = self.fixed_gate_weights.to(dtype=x.dtype)
            gate_weights = gate_weights.unsqueeze(0).expand(x.shape[0], -1)
        else:
            avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            gate_input = self.gate_norm(torch.cat([avg_pool, max_pool], dim=1))
            gate_logits = self.gate(gate_input)
            if self.gate_residual_scale is not None:
                prior = self.gate_prior.to(dtype=gate_logits.dtype)
                bounded_residual = self.gate_residual_scale * torch.tanh(
                    gate_logits
                )
                gate_weights = F.softmax(
                    prior.log() + bounded_residual,
                    dim=-1,
                )
            else:
                gate_weights = F.softmax(
                    gate_logits / self.gate_temperature,
                    dim=-1,
                )

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )
        output = torch.einsum("bk,bkct->bct", gate_weights, expert_outputs)
        if return_gate:
            return output, gate_weights
        return output


class TemporalResidualProjection(nn.Module):
    """Refine temporal evidence while preserving its CLIP-space identity."""

    def __init__(self, dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, features):
        return features + self.mlp(self.norm(features))


class ComponentAwareNaturalAcceptance(nn.Module):
    """Decide when natural pair evidence improves the global C2C decision."""

    feature_dim = 13

    def __init__(self, initial_acceptance=0.2):
        super().__init__()
        if not 0.0 < initial_acceptance < 1.0:
            raise ValueError("initial_acceptance must lie in (0, 1).")
        self.feature_norm = nn.LayerNorm(
            self.feature_dim,
            elementwise_affine=False,
        )
        self.acceptor = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.acceptor.weight)
        nn.init.constant_(
            self.acceptor.bias,
            math.log(
                float(initial_acceptance)
                / (1.0 - float(initial_acceptance))
            ),
        )

    @staticmethod
    def _gather(values, indices):
        return values.gather(1, indices.unsqueeze(1)).squeeze(1)

    @classmethod
    def _candidate_difference(
            cls,
            values,
            proposal_index,
            global_index):
        scale = values.std(
            dim=1,
            unbiased=False,
        ).clamp_min(1.0e-4)
        return (
            cls._gather(values, proposal_index)
            - cls._gather(values, global_index)
        ) / scale

    def forward(
            self,
            global_scores,
            proposal_scores,
            pair_indices,
            verb_logits,
            object_logits,
            global_pair_o_scores,
            global_pair_v_scores,
            verb_condition_scores,
            object_condition_scores):
        global_evidence = global_scores.detach().float()
        proposal_evidence = proposal_scores.detach().float()
        verb_evidence = verb_logits.detach().float()
        object_evidence = object_logits.detach().float()
        pair_o_evidence = global_pair_o_scores.detach().float()
        pair_v_evidence = global_pair_v_scores.detach().float()
        verb_condition_evidence = verb_condition_scores.detach().float()
        object_condition_evidence = object_condition_scores.detach().float()

        global_top, global_indices = global_evidence.topk(2, dim=1)
        proposal_top, proposal_indices = proposal_evidence.topk(2, dim=1)
        global_candidate = global_indices[:, 0]
        proposal_candidate = proposal_indices[:, 0]
        global_scale = global_evidence.std(
            dim=1,
            unbiased=False,
        ).clamp_min(1.0e-4)
        proposal_scale = proposal_evidence.std(
            dim=1,
            unbiased=False,
        ).clamp_min(1.0e-4)

        global_components = pair_indices[global_candidate]
        proposal_components = pair_indices[proposal_candidate]
        global_verb = global_components[:, 0]
        global_object = global_components[:, 1]
        proposal_verb = proposal_components[:, 0]
        proposal_object = proposal_components[:, 1]

        features = torch.stack(
            [
                (global_top[:, 0] - global_top[:, 1]) / global_scale,
                (proposal_top[:, 0] - proposal_top[:, 1])
                / proposal_scale,
                self._candidate_difference(
                    global_evidence,
                    proposal_candidate,
                    global_candidate,
                ),
                self._candidate_difference(
                    proposal_evidence,
                    proposal_candidate,
                    global_candidate,
                ),
                (
                    self._gather(verb_evidence, proposal_verb)
                    - self._gather(verb_evidence, global_verb)
                ) / verb_evidence.std(
                    dim=1,
                    unbiased=False,
                ).clamp_min(1.0e-4),
                (
                    self._gather(object_evidence, proposal_object)
                    - self._gather(object_evidence, global_object)
                ) / object_evidence.std(
                    dim=1,
                    unbiased=False,
                ).clamp_min(1.0e-4),
                self._candidate_difference(
                    verb_condition_evidence,
                    proposal_candidate,
                    global_candidate,
                ),
                self._candidate_difference(
                    object_condition_evidence,
                    proposal_candidate,
                    global_candidate,
                ),
                self._candidate_difference(
                    pair_o_evidence,
                    proposal_candidate,
                    global_candidate,
                ),
                self._candidate_difference(
                    pair_v_evidence,
                    proposal_candidate,
                    global_candidate,
                ),
                proposal_verb.eq(global_verb).float(),
                proposal_object.eq(global_object).float(),
                proposal_candidate.eq(global_candidate).float(),
            ],
            dim=1,
        )
        features = torch.nan_to_num(
            features,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        return torch.sigmoid(
            self.acceptor(self.feature_norm(features))
        ).squeeze(1)


class NaturalPairComponentFeedbackFusion(nn.Module):
    """Decompose natural composition scores into component feedback."""

    def __init__(
            self,
            pair_indices,
            num_verbs,
            num_objects,
            video_dim,
            projection_hidden_dim=128,
            projection_dropout=0.1,
            verb_feedback_strength=0.20,
            object_feedback_strength=0.10,
            fuse_during_training=False,
            decomposition_iterations=8,
            decomposition_ridge=1.0e-3):
        super().__init__()
        if verb_feedback_strength < 0.0 or object_feedback_strength < 0.0:
            raise ValueError(
                "Component feedback strengths must be non-negative."
            )
        if int(decomposition_iterations) <= 0:
            raise ValueError("decomposition_iterations must be positive.")
        if float(decomposition_ridge) < 0.0:
            raise ValueError("decomposition_ridge must be non-negative.")
        pair_indices = torch.as_tensor(pair_indices, dtype=torch.long)
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("pair_indices must have shape [num_pairs, 2].")
        if pair_indices.shape[0] == 0:
            raise ValueError("pair_indices must not be empty.")
        self.register_buffer("pair_indices", pair_indices)
        self.num_verbs = int(num_verbs)
        self.num_objects = int(num_objects)
        self.verb_feedback_strength = float(verb_feedback_strength)
        self.object_feedback_strength = float(object_feedback_strength)
        self.fuse_during_training = bool(fuse_during_training)
        self.decomposition_iterations = int(decomposition_iterations)
        self.decomposition_ridge = float(decomposition_ridge)
        self.register_buffer(
            "verb_pair_counts",
            torch.bincount(
                pair_indices[:, 0],
                minlength=self.num_verbs,
            ).float(),
            persistent=False,
        )
        self.register_buffer(
            "object_pair_counts",
            torch.bincount(
                pair_indices[:, 1],
                minlength=self.num_objects,
            ).float(),
            persistent=False,
        )
        self.temporal_projection = TemporalResidualProjection(
            int(video_dim),
            hidden_dim=int(projection_hidden_dim),
            dropout=float(projection_dropout),
        )

    def _valid_pair_scores(self, scores):
        return scores[
            :,
            self.pair_indices[:, 0],
            self.pair_indices[:, 1],
        ]

    def _scatter_valid_evidence(self, evidence):
        flat_indices = (
            self.pair_indices[:, 0] * self.num_objects
            + self.pair_indices[:, 1]
        )
        full_evidence = evidence.new_zeros(
            evidence.shape[0],
            self.num_verbs * self.num_objects,
        )
        full_evidence = full_evidence.scatter(
            1,
            flat_indices.unsqueeze(0).expand(evidence.shape[0], -1),
            evidence,
        )
        return full_evidence.view(
            evidence.shape[0],
            self.num_verbs,
            self.num_objects,
        )

    def _component_decomposition(self, scores):
        """Backfit additive verb and object effects on the valid-pair graph."""
        batch_size, pair_count = scores.shape
        pair_verbs = self.pair_indices[:, 0]
        pair_objects = self.pair_indices[:, 1]
        expanded_verbs = pair_verbs.unsqueeze(0).expand(batch_size, -1)
        expanded_objects = pair_objects.unsqueeze(0).expand(batch_size, -1)
        verb_counts = self.verb_pair_counts.to(
            device=scores.device,
            dtype=scores.dtype,
        )
        object_counts = self.object_pair_counts.to(
            device=scores.device,
            dtype=scores.dtype,
        )
        verb_denominator = (
            verb_counts + self.decomposition_ridge
        ).clamp_min(1.0)
        object_denominator = (
            object_counts + self.decomposition_ridge
        ).clamp_min(1.0)

        grand_mean = scores.mean(dim=1, keepdim=True)
        verb_effect = scores.new_zeros(batch_size, self.num_verbs)
        object_effect = scores.new_zeros(batch_size, self.num_objects)
        pair_count_float = float(pair_count)

        for _ in range(self.decomposition_iterations):
            verb_residual = (
                scores
                - grand_mean
                - object_effect.gather(1, expanded_objects)
            )
            verb_sums = scores.new_zeros(batch_size, self.num_verbs)
            verb_sums.scatter_add_(1, expanded_verbs, verb_residual)
            verb_effect = verb_sums / verb_denominator.unsqueeze(0)
            verb_offset = (
                verb_effect * verb_counts.unsqueeze(0)
            ).sum(dim=1, keepdim=True) / pair_count_float
            verb_effect = verb_effect - verb_offset
            grand_mean = grand_mean + verb_offset

            object_residual = (
                scores
                - grand_mean
                - verb_effect.gather(1, expanded_verbs)
            )
            object_sums = scores.new_zeros(batch_size, self.num_objects)
            object_sums.scatter_add_(
                1,
                expanded_objects,
                object_residual,
            )
            object_effect = object_sums / object_denominator.unsqueeze(0)
            object_offset = (
                object_effect * object_counts.unsqueeze(0)
            ).sum(dim=1, keepdim=True) / pair_count_float
            object_effect = object_effect - object_offset
            grand_mean = grand_mean + object_offset

        return verb_effect, object_effect

    @staticmethod
    def _unit_scale(component_feedback, global_std):
        """Center and rescale one component to the global row deviation."""
        centered = (
            component_feedback
            - component_feedback.mean(dim=1, keepdim=True)
        )
        component_std = centered.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1.0e-4)
        return torch.nan_to_num(
            centered * (global_std / component_std),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def forward(
            self,
            global_pair_scores,
            temporal_video_embedding,
            natural_pair_text_features):
        if natural_pair_text_features.shape[0] != self.pair_indices.shape[0]:
            raise ValueError(
                "Natural text bank and valid pair list have different sizes."
            )
        source_dtype = global_pair_scores.dtype
        with torch.cuda.amp.autocast(enabled=False):
            global_reference = global_pair_scores.float()
            if self.training:
                global_reference = global_reference.detach()
            temporal_features = F.normalize(
                self.temporal_projection(
                    temporal_video_embedding.float()
                ),
                dim=-1,
                eps=1.0e-6,
            )
            natural_text = F.normalize(
                natural_pair_text_features.detach().float(),
                dim=-1,
                eps=1.0e-6,
            )
            natural_scores = temporal_features @ natural_text.t()
            global_valid = self._valid_pair_scores(global_reference)

            verb_effect, object_effect = self._component_decomposition(
                natural_scores
            )
            pair_verbs = self.pair_indices[:, 0]
            pair_objects = self.pair_indices[:, 1]
            verb_pair_feedback = verb_effect[:, pair_verbs]
            object_pair_feedback = object_effect[:, pair_objects]
            global_std = global_valid.detach().std(
                dim=1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1.0e-4)
            # Unit evidence: one global-row-deviation worth of each component.
            # The training path fuses with the configured strengths while the
            # evaluation path can recompose with any (alpha_v, alpha_o).
            verb_unit_evidence = self._unit_scale(
                verb_pair_feedback,
                global_std,
            )
            object_unit_evidence = self._unit_scale(
                object_pair_feedback,
                global_std,
            )
            natural_evidence = (
                self.verb_feedback_strength * verb_unit_evidence
                + self.object_feedback_strength * object_unit_evidence
            )

            proposal_evidence_full = self._scatter_valid_evidence(
                natural_evidence
            )
            proposal_scores = (
                global_reference + proposal_evidence_full
            )
            # Train without score fusion so the evidence branch never
            # pollutes the global trunk path; inference applies the fixed
            # (verb, object) strengths once.
            apply_fusion = (
                self.fuse_during_training or (not self.training)
            )
            corrected_scores = (
                proposal_scores if apply_fusion else global_reference
            )
            verb_unit_evidence_full = self._scatter_valid_evidence(
                verb_unit_evidence
            )
            object_unit_evidence_full = self._scatter_valid_evidence(
                object_unit_evidence
            )
            top1_agreement = global_valid.argmax(dim=1).eq(
                natural_evidence.argmax(dim=1)
            ).float()
            applied_strength = (
                self.verb_feedback_strength + self.object_feedback_strength
                if apply_fusion else 0.0
            )
            fusion_weight = natural_evidence.new_full(
                (natural_evidence.shape[0],),
                applied_strength,
            )
            evidence_scale = global_std
            diagnostics = torch.stack(
                [
                    natural_scores.std(dim=1, unbiased=False),
                    natural_evidence.abs().mean(dim=1),
                    top1_agreement,
                    fusion_weight,
                    verb_pair_feedback.abs().mean(dim=1),
                    object_pair_feedback.abs().mean(dim=1),
                ],
                dim=1,
            )
            component_feedback_scores = torch.cat(
                [verb_effect, object_effect],
                dim=1,
            )

        return (
            corrected_scores.to(dtype=source_dtype),
            proposal_scores.to(dtype=source_dtype),
            diagnostics,
            evidence_scale.squeeze(1),
            fusion_weight,
            component_feedback_scores,
            verb_unit_evidence_full.to(dtype=source_dtype),
            object_unit_evidence_full.to(dtype=source_dtype),
        )


class ObjectAnchoredTemporalTransfer(nn.Module):
    """Transfer temporal action relations without replacing object evidence."""

    feature_dim = 7

    def __init__(
            self,
            pair_object_indices,
            num_objects,
            max_weight=0.7,
            initial_acceptance=0.25):
        super().__init__()
        if not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must lie in (0, 1].")
        if not 0.0 < initial_acceptance < 1.0:
            raise ValueError("initial_acceptance must lie in (0, 1).")

        self.num_objects = int(num_objects)
        self.max_weight = float(max_weight)
        pair_object_indices = torch.as_tensor(
            pair_object_indices,
            dtype=torch.long,
        )
        self.register_buffer(
            "pair_object_indices",
            pair_object_indices,
        )
        object_groups = [
            pair_object_indices.eq(object_index).nonzero(
                as_tuple=False
            ).flatten()
            for object_index in range(self.num_objects)
        ]
        max_group_size = max(
            max((group.numel() for group in object_groups), default=0),
            1,
        )
        padded_pair_indices = torch.zeros(
            self.num_objects,
            max_group_size,
            dtype=torch.long,
        )
        group_valid_mask = torch.zeros(
            self.num_objects,
            max_group_size,
            dtype=torch.bool,
        )
        pair_slot_indices = torch.zeros_like(pair_object_indices)
        for object_index, group in enumerate(object_groups):
            group_size = group.numel()
            if group_size == 0:
                continue
            padded_pair_indices[object_index, :group_size] = group
            group_valid_mask[object_index, :group_size] = True
            pair_slot_indices[group] = torch.arange(group_size)
        self.register_buffer("padded_pair_indices", padded_pair_indices)
        self.register_buffer("group_valid_mask", group_valid_mask)
        self.register_buffer("pair_slot_indices", pair_slot_indices)
        self.feature_norm = nn.LayerNorm(
            self.feature_dim,
            elementwise_affine=False,
        )
        self.utility_classifier = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.utility_classifier.weight)
        nn.init.constant_(
            self.utility_classifier.bias,
            math.log(initial_acceptance / (1.0 - initial_acceptance)),
        )

    @staticmethod
    def _masked_scale(values, valid_mask, eps):
        valid = valid_mask.to(dtype=values.dtype)
        count = valid.sum(dim=-1).clamp_min(1.0)
        mean = (values * valid).sum(dim=-1) / count
        variance = (
            (values - mean.unsqueeze(-1)).square() * valid
        ).sum(dim=-1) / count
        return (variance + float(eps) ** 2).sqrt()

    @staticmethod
    def _masked_top2(values, valid_mask):
        minimum = values.new_tensor(-1.0e4)
        masked_values = values.masked_fill(~valid_mask, minimum)
        top_count = min(2, values.shape[-1])
        top_values, top_indices = masked_values.topk(top_count, dim=-1)
        top_value = top_values[..., 0]
        top_index = top_indices[..., 0]
        if top_count == 1:
            return top_value, top_value, top_index
        valid_count = valid_mask.sum(dim=-1)
        second_value = torch.where(
            valid_count > 1,
            top_values[..., 1],
            top_value,
        )
        return top_value, second_value, top_index

    def _utility_features(
            self,
            global_relation,
            temporal_relation,
            valid_mask):
        eps = 1.0e-4
        global_scale = self._masked_scale(
            global_relation,
            valid_mask,
            eps,
        )
        temporal_scale = self._masked_scale(
            temporal_relation,
            valid_mask,
            eps,
        )
        global_top, global_second, global_index = self._masked_top2(
            global_relation,
            valid_mask,
        )
        temporal_top, temporal_second, temporal_index = self._masked_top2(
            temporal_relation,
            valid_mask,
        )
        global_at_temporal = global_relation.gather(
            -1,
            temporal_index.unsqueeze(-1),
        ).squeeze(-1)
        temporal_at_global = temporal_relation.gather(
            -1,
            global_index.unsqueeze(-1),
        ).squeeze(-1)
        valid = valid_mask.to(dtype=global_relation.dtype)
        relation_dot = (
            global_relation * temporal_relation * valid
        ).sum(dim=-1)
        global_norm = (
            (global_relation.square() * valid).sum(dim=-1) + eps ** 2
        ).sqrt()
        temporal_norm = (
            (temporal_relation.square() * valid).sum(dim=-1) + eps ** 2
        ).sqrt()
        relation_cosine = relation_dot / (
            global_norm * temporal_norm
        ).clamp_min(eps)
        relation_difference = (
            (temporal_relation - global_relation).abs() * valid
        ).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        features = torch.stack(
            [
                (global_top - global_second) / global_scale,
                (temporal_top - temporal_second) / temporal_scale,
                (global_at_temporal - global_top) / global_scale,
                (temporal_top - temporal_at_global) / temporal_scale,
                global_index.eq(temporal_index).float(),
                relation_cosine,
                relation_difference / global_scale,
            ],
            dim=-1,
        )
        modifiable = valid_mask.sum(dim=-1) > 1
        features = features.masked_fill(
            ~modifiable.unsqueeze(-1),
            0.0,
        )
        return torch.nan_to_num(
            features,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def forward(self, global_scores, temporal_scores):
        if global_scores.shape != temporal_scores.shape:
            raise ValueError(
                "global_scores and temporal_scores must have equal shapes."
            )
        if global_scores.shape[1] != self.pair_object_indices.numel():
            raise ValueError(
                "Pair score width does not match pair_object_indices."
            )

        source_dtype = global_scores.dtype
        global_reference = global_scores.detach().float()
        temporal_reference = temporal_scores.float()
        batch_size = global_reference.shape[0]
        group_shape = (
            batch_size,
            self.num_objects,
            self.padded_pair_indices.shape[1],
        )
        gather_indices = self.padded_pair_indices.flatten().unsqueeze(0)
        gather_indices = gather_indices.expand(batch_size, -1)
        global_group = global_reference.gather(
            1,
            gather_indices,
        ).view(group_shape)
        temporal_group = temporal_reference.gather(
            1,
            gather_indices,
        ).view(group_shape)
        valid_mask = self.group_valid_mask.unsqueeze(0)
        minimum = global_reference.new_tensor(-1.0e4)
        global_masked = global_group.masked_fill(~valid_mask, minimum)
        temporal_masked = temporal_group.masked_fill(~valid_mask, minimum)
        object_anchor = global_masked.max(dim=-1, keepdim=True).values
        temporal_anchor = temporal_masked.max(dim=-1, keepdim=True).values
        valid_object = valid_mask.any(dim=-1, keepdim=True)
        object_anchor = torch.where(
            valid_object,
            object_anchor,
            torch.zeros_like(object_anchor),
        )
        temporal_anchor = torch.where(
            valid_object,
            temporal_anchor,
            torch.zeros_like(temporal_anchor),
        )
        global_relation = (global_group - object_anchor).masked_fill(
            ~valid_mask,
            0.0,
        )
        temporal_relation = (temporal_group - temporal_anchor).masked_fill(
            ~valid_mask,
            0.0,
        )
        global_scale = self._masked_scale(
            global_relation,
            valid_mask,
            1.0e-4,
        ).unsqueeze(-1)
        temporal_scale = self._masked_scale(
            temporal_relation,
            valid_mask,
            1.0e-4,
        ).unsqueeze(-1)
        scale_ratio = (
            global_scale / temporal_scale
        ).detach().clamp(min=0.25, max=4.0)
        temporal_relation = (
            temporal_relation * scale_ratio
        ).masked_fill(~valid_mask, 0.0)

        proposal_group = (
            object_anchor + temporal_relation
        ).masked_fill(~valid_mask, 0.0)
        features = self._utility_features(
            global_relation.detach(),
            temporal_relation.detach(),
            valid_mask,
        )
        transfer_logits = self.utility_classifier(
            self.feature_norm(features)
        ).squeeze(-1)
        modifiable = self.group_valid_mask.sum(dim=-1) > 1
        transfer_weights = self.max_weight * torch.sigmoid(
            transfer_logits
        )
        transfer_weights = transfer_weights * modifiable.unsqueeze(0)
        mixed_relation = global_relation + transfer_weights.unsqueeze(-1) * (
            temporal_relation - global_relation
        )
        mixed_max = mixed_relation.masked_fill(
            ~valid_mask,
            minimum,
        ).max(dim=-1, keepdim=True).values
        mixed_relation = (mixed_relation - mixed_max).masked_fill(
            ~valid_mask,
            0.0,
        )
        corrected_group = (
            object_anchor + mixed_relation
        ).masked_fill(~valid_mask, 0.0)

        pair_objects = self.pair_object_indices
        pair_slots = self.pair_slot_indices
        proposal_scores = proposal_group[:, pair_objects, pair_slots]
        corrected_scores = corrected_group[:, pair_objects, pair_slots]

        correction = corrected_scores - global_reference
        return (
            corrected_scores.to(dtype=source_dtype),
            proposal_scores.to(dtype=source_dtype),
            correction.to(dtype=source_dtype),
            transfer_logits,
            transfer_weights,
        )


class GlobalAnchoredTemporalResidual(nn.Module):
    """Add a bounded temporal residual without modifying global logits."""

    def __init__(self, num_objects, max_scale=0.30, initial_scale=0.10):
        super().__init__()
        if max_scale <= 0.0:
            raise ValueError("max_scale must be positive.")
        if not 0.0 < initial_scale < max_scale:
            raise ValueError("initial_scale must lie in (0, max_scale).")
        self.num_objects = int(num_objects)
        self.max_scale = float(max_scale)
        initial_ratio = float(initial_scale) / self.max_scale
        self.scale_logit = nn.Parameter(torch.tensor(
            math.log(initial_ratio / (1.0 - initial_ratio)),
            dtype=torch.float32,
        ))

    def current_scale(self):
        return self.max_scale * torch.sigmoid(self.scale_logit)

    def forward(self, global_scores, temporal_scores):
        if global_scores.shape != temporal_scores.shape:
            raise ValueError(
                "global_scores and temporal_scores must have equal shapes."
            )
        global_reference = global_scores.detach().float()
        temporal_reference = temporal_scores.float()
        temporal_residual = (
            temporal_reference - temporal_reference.mean(dim=1, keepdim=True)
        )
        residual_scale = self.current_scale()
        proposal_scores = global_reference + temporal_residual
        corrected_scores = global_reference + residual_scale * temporal_residual

        batch_size = global_scores.shape[0]
        scale_logits = self.scale_logit.expand(batch_size, self.num_objects)
        scale_weights = residual_scale.expand(batch_size, self.num_objects)
        correction = corrected_scores - global_reference
        return (
            corrected_scores.to(dtype=global_scores.dtype),
            proposal_scores.to(dtype=global_scores.dtype),
            correction.to(dtype=global_scores.dtype),
            scale_logits,
            scale_weights,
        )


class MLP_ST(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(
            self,
            inp_dim,
            out_dim,
            num_layers=1,
            relu=True,
            bias=True,
            dropout=False,
            norm=False,
            layers=[],
            use_temporal_experts=False,
            temporal_expert_kernel_sizes=(3, 5, 7, 9),
            temporal_expert_temperature=1.0,
            temporal_expert_gate_norm=False,
            temporal_expert_gate_prior=None):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            if use_temporal_experts:
                mod.append(TemporalExpertConv1d(
                    incoming,
                    outgoing,
                    kernel_sizes=temporal_expert_kernel_sizes,
                    bias=bias,
                    temperature=temporal_expert_temperature,
                    use_gate_norm=temporal_expert_gate_norm,
                    gate_prior=temporal_expert_gate_prior,
                ))
            else:
                mod.append(nn.Conv1d(
                    incoming,
                    outgoing,
                    kernel_size=3,
                    bias=bias,
                    padding=1,
                ))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        # mod.append(nn.Linear(incoming, out_dim, bias=bias))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        full_attention_mask = next(
            block.attn_mask
            for block in self.transformer.resblocks
            if block.attn_mask is not None
        )
        self.register_buffer(
            "full_attention_mask",
            full_attention_mask.detach().clone(),
            persistent=False,
        )
        self.dtype = clip_model.dtype

    def _set_attention_length(self, sequence_length, device):
        if sequence_length > self.full_attention_mask.shape[0]:
            raise ValueError(
                "Text sequence length {} exceeds CLIP capacity {}.".format(
                    sequence_length,
                    self.full_attention_mask.shape[0],
                )
            )
        attention_mask = self.full_attention_mask[
            :sequence_length,
            :sequence_length,
        ].to(device=device)
        for block in self.transformer.resblocks:
            block.attn_mask = attention_mask

    def forward(self, x, tokenized_prompts):  # have been added positional emb
        self._set_attention_length(x.shape[1], x.device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_token_ids(self, tokenized_prompts):
        sequence_length = tokenized_prompts.shape[1]
        token_embeddings = self.token_embedding(
            tokenized_prompts
        ).to(dtype=self.dtype)
        positional = self.positional_embedding[
            :sequence_length
        ].to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        return self.forward(
            token_embeddings + positional.unsqueeze(0),
            tokenized_prompts,
        )


class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj

        self.num_frames=cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)#be annotated


        return out


class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        """
        Using component to deduce the composition, without composition
        """
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        # self.dtype = clip_model.dtype

        # ======== C2C part =====
        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = []
        for a in fc_emb:
            a = int(a)
            layers.append(a)

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False,
                           norm=True, layers=layers)

        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False,
                           norm=True, layers=layers)

        kernel_sizes = getattr(
            cfg,
            'temporal_composition_kernel_sizes',
            '3,5,7,9',
        )
        if isinstance(kernel_sizes, str):
            kernel_sizes = tuple(
                int(value.strip())
                for value in kernel_sizes.split(',')
                if value.strip()
            )
        else:
            kernel_sizes = tuple(int(value) for value in kernel_sizes)
        temporal_composition_temperature = float(getattr(
            cfg,
            'temporal_composition_temperature',
            2.5,
        ))
        temporal_composition_gate_norm = bool(getattr(
            cfg, 'temporal_composition_gate_norm', False
        ))
        gate_prior = getattr(
            cfg,
            'temporal_composition_gate_prior',
            '0.1,0.1,0.4,0.4',
        )
        if isinstance(gate_prior, str):
            gate_prior = tuple(
                float(value.strip())
                for value in gate_prior.split(',')
                if value.strip()
            )
        else:
            gate_prior = tuple(float(value) for value in gate_prior)
        fixed_gate_weights = getattr(
            cfg,
            'temporal_composition_fixed_gate_weights',
            None,
        )
        if isinstance(fixed_gate_weights, str):
            if fixed_gate_weights.strip().lower() in ('', 'none', 'null'):
                fixed_gate_weights = None
            else:
                fixed_gate_weights = tuple(
                    float(value.strip())
                    for value in fixed_gate_weights.split(',')
                    if value.strip()
                )
        elif fixed_gate_weights is not None:
            fixed_gate_weights = tuple(
                float(value) for value in fixed_gate_weights
            )
        gate_residual_scale = getattr(
            cfg,
            'temporal_composition_gate_residual_scale',
            None,
        )
        if isinstance(gate_residual_scale, str):
            if gate_residual_scale.strip().lower() in ('', 'none', 'null'):
                gate_residual_scale = None
            else:
                gate_residual_scale = float(gate_residual_scale)
        elif gate_residual_scale is not None:
            gate_residual_scale = float(gate_residual_scale)

        self.c2c_VE1 = MLP_ST(
            cfg.feat_dim,
            int(cfg.emb_dim),
            relu=cfg.relu,
            num_layers=cfg.nlayers,
            dropout=False,
            norm=True,
            layers=layers,
            use_temporal_experts=False,
        )

        self.c2c_VE2 = MLP_ST(
            cfg.feat_dim,
            int(cfg.emb_dim),
            relu=cfg.relu,
            num_layers=cfg.nlayers,
            dropout=False,
            norm=True,
            layers=layers,
            use_temporal_experts=False,
        )


        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)  # TODO
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)  # TODO

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)  # TODO
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)  # TODO

        self.use_temporal_composition_branch = bool(getattr(
            cfg, 'temporal_composition_enabled', True
        ))
        self.temporal_composition_feature_residual = float(getattr(
            cfg, 'temporal_composition_feature_residual', 0.05
        ))
        self.temporal_composition_detach_backbone = bool(getattr(
            cfg, 'temporal_composition_detach_backbone', False
        ))
        backbone_gradient_scale = getattr(
            cfg, 'temporal_composition_backbone_gradient_scale', None
        )
        if isinstance(backbone_gradient_scale, str):
            stripped = backbone_gradient_scale.strip().lower()
            backbone_gradient_scale = (
                None if stripped in ('', 'none', 'null')
                else float(stripped)
            )
        if backbone_gradient_scale is None:
            backbone_gradient_scale = (
                0.0 if self.temporal_composition_detach_backbone else 1.0
            )
        self.temporal_composition_backbone_gradient_scale = float(
            backbone_gradient_scale
        )
        if not 0.0 <= self.temporal_composition_backbone_gradient_scale <= 1.0:
            raise ValueError(
                "temporal_composition_backbone_gradient_scale must lie "
                "in [0, 1]."
            )
        self.natural_pair_context_length = int(getattr(
            cfg, 'natural_pair_context_length', 48
        ))
        self.natural_pair_text_batch_size = int(getattr(
            cfg, 'natural_pair_text_batch_size', 256
        ))
        self.natural_pair_verb_feedback_strength = float(getattr(
            cfg, 'natural_pair_verb_feedback_strength', 0.20
        ))
        self.natural_pair_object_feedback_strength = float(getattr(
            cfg, 'natural_pair_object_feedback_strength', 0.10
        ))
        self.natural_pair_fuse_during_training = bool(getattr(
            cfg, 'natural_pair_fuse_during_training', False
        ))
        self.temporal_relation_shaper_enabled = bool(getattr(
            cfg, 'temporal_relation_shaper_enabled', False
        ))
        self.natural_pair_decomposition_iterations = int(getattr(
            cfg, 'natural_pair_decomposition_iterations', 8
        ))
        self.natural_pair_decomposition_ridge = float(getattr(
            cfg, 'natural_pair_decomposition_ridge', 1.0e-3
        ))
        prompt_weights = getattr(
            cfg,
            'natural_pair_prompt_weights',
            '0.25,0.25,0.50',
        )
        if isinstance(prompt_weights, str):
            prompt_weights = tuple(
                float(value.strip())
                for value in prompt_weights.split(',')
                if value.strip()
            )
        else:
            prompt_weights = tuple(float(value) for value in prompt_weights)
        if not prompt_weights or any(weight < 0.0 for weight in prompt_weights):
            raise ValueError(
                "natural_pair_prompt_weights must be non-negative."
            )
        weight_sum = float(sum(prompt_weights))
        if weight_sum <= 0.0:
            raise ValueError(
                "natural_pair_prompt_weights must have a positive sum."
            )
        self.natural_pair_prompt_weights = tuple(
            weight / weight_sum for weight in prompt_weights
        )
        if self.temporal_composition_feature_residual < 0.0:
            raise ValueError(
                "temporal_composition_feature_residual must be non-negative."
            )
        if not 1 <= self.natural_pair_context_length <= 77:
            raise ValueError(
                "natural_pair_context_length must lie in [1, 77]."
            )
        if self.natural_pair_text_batch_size <= 0:
            raise ValueError("natural_pair_text_batch_size must be positive.")

        self.use_gse = getattr(cfg, 'use_gse', False)
        self.gse_position = getattr(cfg, 'gse_position', 'pre_fusion_cond_only')
        if self.use_gse and self.gse_position != 'pre_fusion_cond_only':
            raise ValueError(
                "This experiment only supports gse_position='pre_fusion_cond_only'."
            )
        if self.use_gse:
            self.c2c_gse_cond = GlobalSpatialExpert(
                dim=cfg.emb_dim,
                num_heads=getattr(cfg, 'gse_heads', 4),
                dropout=getattr(cfg, 'gse_dropout', 0.1),
                alpha_init=getattr(cfg, 'gse_alpha_init', 0.05),
                beta_init=getattr(cfg, 'gse_beta_init', 0.05),
            )
        self.use_mmn_spatial_refiner = getattr(cfg, 'use_mmn_spatial_refiner', False)
        if self.use_mmn_spatial_refiner:
            self.c2c_mmn_spatial_refiner = MMNCrossRoutedSpatialRefiner(
                num_object_tokens=len(train_dataset.objs),
                num_verb_tokens=len(train_dataset.attrs),
                dim=cfg.emb_dim,
                bottleneck_dim=getattr(cfg, 'mmn_spatial_bottleneck_dim', 128),
                dropout=getattr(cfg, 'mmn_spatial_dropout', 0.15),
                residual_init=getattr(cfg, 'mmn_spatial_residual_init', 0.05),
            )

        # Preserve the legacy global trunk's initialization and RNG trajectory.
        if self.use_temporal_composition_branch:
            branch_seed = int(getattr(cfg, 'seed', 0)) + int(getattr(
                cfg, 'temporal_composition_seed_offset', 104729
            ))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(branch_seed)
                self.c2c_temporal_composition_expert = TemporalExpertConv1d(
                    cfg.feat_dim,
                    cfg.feat_dim,
                    kernel_sizes=kernel_sizes,
                    temperature=temporal_composition_temperature,
                    use_gate_norm=temporal_composition_gate_norm,
                    gate_prior=gate_prior,
                    fixed_gate_weights=fixed_gate_weights,
                    gate_residual_scale=gate_residual_scale,
                )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(branch_seed + 1)
                pair_indices = torch.tensor(
                    [
                        (
                            train_dataset.attr2idx[attr],
                            train_dataset.obj2idx[obj],
                        )
                        for attr, obj in train_dataset.pairs
                    ],
                    dtype=torch.long,
                )
                self.c2c_natural_pair_probabilistic_fusion = (
                    NaturalPairComponentFeedbackFusion(
                        pair_indices=pair_indices,
                        num_verbs=len(train_dataset.attrs),
                        num_objects=len(train_dataset.objs),
                        video_dim=cfg.feat_dim,
                        projection_hidden_dim=int(getattr(
                            cfg,
                            'natural_pair_projection_hidden_dim',
                            128,
                        )),
                        projection_dropout=float(getattr(
                            cfg,
                            'natural_pair_projection_dropout',
                            0.1,
                        )),
                        verb_feedback_strength=(
                            self.natural_pair_verb_feedback_strength
                        ),
                        object_feedback_strength=(
                            self.natural_pair_object_feedback_strength
                        ),
                        fuse_during_training=(
                            self.natural_pair_fuse_during_training
                        ),
                        decomposition_iterations=(
                            self.natural_pair_decomposition_iterations
                        ),
                        decomposition_ridge=(
                            self.natural_pair_decomposition_ridge
                        ),
                    )
                )
            if self.temporal_relation_shaper_enabled:
                # Training-only trunk shaper reproducing run "1"'s proven
                # mechanism: a temporal expert view whose pair-level CE flows
                # back into the shared trunk and text heads. Its scores are
                # never fused at evaluation time.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(branch_seed + 2)
                    self.c2c_temporal_relation_shaper = MLP_ST(
                        cfg.feat_dim,
                        int(cfg.emb_dim),
                        relu=cfg.relu,
                        num_layers=cfg.nlayers,
                        dropout=False,
                        norm=True,
                        layers=layers,
                        use_temporal_experts=True,
                        temporal_expert_kernel_sizes=kernel_sizes,
                        temporal_expert_temperature=(
                            temporal_composition_temperature
                        ),
                        temporal_expert_gate_norm=(
                            temporal_composition_gate_norm
                        ),
                        temporal_expert_gate_prior=gate_prior,
                    )
                    self.c2c_temporal_shaper_projection = (
                        TemporalResidualProjection(
                            int(cfg.emb_dim),
                            hidden_dim=int(getattr(
                                cfg,
                                'temporal_shaper_projection_hidden_dim',
                                128,
                            )),
                            dropout=float(getattr(
                                cfg,
                                'temporal_shaper_projection_dropout',
                                0.1,
                            )),
                        )
                    )
                train_pair_indices = torch.tensor(
                    [
                        (
                            train_dataset.attr2idx[attr],
                            train_dataset.obj2idx[obj],
                        )
                        for attr, obj in train_dataset.train_pairs
                    ],
                    dtype=torch.long,
                )
                self.register_buffer(
                    'temporal_train_pair_indices',
                    train_pair_indices,
                    persistent=False,
                )
            natural_prompts = []
            for attr, obj in train_dataset.pairs:
                natural_prompts.extend(
                    build_role_robust_natural_prompts(attr, obj)
                )
            natural_tokens = clip.tokenize(
                natural_prompts,
                context_length=self.natural_pair_context_length,
                truncate=True,
            )
            self.register_buffer(
                "natural_pair_token_ids",
                natural_tokens,
                persistent=False,
            )
            self.register_buffer(
                "natural_pair_text_features",
                torch.empty(0, cfg.feat_dim),
                persistent=False,
            )
            self.natural_pair_prompt_count = len(
                build_role_robust_natural_prompts(
                    "pick [something]",
                    "object",
                )
            )
            if (
                len(self.natural_pair_prompt_weights)
                != self.natural_pair_prompt_count
            ):
                raise ValueError(
                    "natural_pair_prompt_weights length must match the "
                    "number of natural prompts per pair "
                    "(%d)." % self.natural_pair_prompt_count
                )

    @torch.no_grad()
    def prepare_natural_pair_text_bank(self):
        if not self.use_temporal_composition_branch:
            return
        if self.natural_pair_text_features.numel() > 0:
            return
        encoded_batches = []
        for start in range(
                0,
                self.natural_pair_token_ids.shape[0],
                self.natural_pair_text_batch_size):
            token_batch = self.natural_pair_token_ids[
                start:start + self.natural_pair_text_batch_size
            ]
            encoded_batches.append(
                self.text_encoder.encode_token_ids(token_batch).float()
            )
        encoded = torch.cat(encoded_batches, dim=0)
        pair_count = (
            self.c2c_natural_pair_probabilistic_fusion
            .pair_indices.shape[0]
        )
        encoded = encoded.view(
            pair_count,
            self.natural_pair_prompt_count,
            -1,
        )
        encoded = F.normalize(encoded, dim=-1)
        prompt_weights = encoded.new_tensor(
            self.natural_pair_prompt_weights
        ).view(1, self.natural_pair_prompt_count, 1)
        encoded = (encoded * prompt_weights).sum(dim=1)
        self.natural_pair_text_features = F.normalize(
            encoded,
            dim=-1,
            eps=1.0e-6,
        )

    @staticmethod
    def _row_standardize(scores):
        centered = scores - scores.mean(dim=1, keepdim=True)
        return centered / centered.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1.0e-4)

    def _temporal_relation_shaper_scores(
            self,
            video_features,
            verb_text_features,
            obj_text_features,
            obj_logits,
            global_pair_scores):
        """Run "1"-style trunk shaper: pair CE through a temporal view.

        Gradients intentionally reach the shared video encoder and the text
        heads; the returned scores are only consumed by the training loss.
        """
        shaper_features = self.c2c_temporal_relation_shaper(video_features)
        shaper_embedding = self.c2c_temporal_shaper_projection(
            shaper_features.mean(dim=-1)
        )
        shaper_embedding = F.normalize(shaper_embedding, dim=-1)

        verb_text = F.normalize(verb_text_features, dim=-1)
        obj_text = F.normalize(obj_text_features, dim=-1)
        train_pairs = self.temporal_train_pair_indices
        pair_text = F.normalize(
            verb_text[train_pairs[:, 0]] + obj_text[train_pairs[:, 1]],
            dim=-1,
        )
        pair_scores = shaper_embedding @ pair_text.t()

        # Remove direct object evidence and restore object identity from the
        # global component classifier, so the shaper contributes
        # action/relation supervision instead of a second object classifier.
        object_scores = shaper_embedding @ obj_text.t()
        relation_scores = (
            pair_scores - object_scores[:, train_pairs[:, 1]]
        )
        object_anchor = obj_logits[:, train_pairs[:, 1]].detach()

        proposal_shape = self._row_standardize(relation_scores.float())
        proposal_shape = proposal_shape + self._row_standardize(
            object_anchor.float()
        )
        proposal_shape = self._row_standardize(proposal_shape)

        global_train = global_pair_scores[
            :, train_pairs[:, 0], train_pairs[:, 1]
        ].detach().float()
        global_mean = global_train.mean(dim=1, keepdim=True)
        global_std = global_train.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1.0e-4)
        return (
            proposal_shape * global_std + global_mean
        ).to(dtype=global_pair_scores.dtype)

    def forward(
            self,
            video,
            pairs=None,
            return_branch_scores=False):
        verb_prompts = self.verb_prompt_learner()
        verb_clip_features = self.text_encoder(
            verb_prompts, self.verb_tokenized_prompts
        )
        verb_text_features = self.c2c_text_v(verb_clip_features)

        obj_prompts = self.obj_prompt_learner()
        obj_clip_features = self.text_encoder(
            obj_prompts, self.obj_tokenized_prompts
        )
        obj_text_features = self.c2c_text_o(obj_clip_features)

        video_features = self.video_encoder(video)# b d t

        temporal_video_embedding = None
        temporal_gate_weights = None
        if self.use_temporal_composition_branch:
            gradient_scale = (
                self.temporal_composition_backbone_gradient_scale
            )
            if not self.training or gradient_scale <= 0.0:
                temporal_branch_input = video_features.detach()
            elif gradient_scale >= 1.0:
                temporal_branch_input = video_features
            else:
                # Same values as video_features; only a scaled fraction of
                # the temporal-branch gradient reaches the shared backbone.
                temporal_branch_input = (
                    gradient_scale * video_features
                    + (1.0 - gradient_scale) * video_features.detach()
                )
            temporal_delta, temporal_gate_weights = (
                self.c2c_temporal_composition_expert(
                    temporal_branch_input, return_gate=True
                )
            )
            temporal_refined_features = temporal_branch_input.float() + (
                self.temporal_composition_feature_residual
                * temporal_delta.float()
            )
            temporal_video_embedding = temporal_refined_features.mean(
                dim=-1
            )

        # independent learning
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))  # b,c
        v_feat_t = self.c2c_VE1(video_features)  # b,c,t
        o_feat_normed = F.normalize(o_feat, dim=1)
        v_feat = v_feat_t.mean(dim=-1)  # b,c
        v_feat_normed = F.normalize(v_feat, dim=1)

        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        verb_text_features_norm = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)
        obj_text_features_norm = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)

        verb_logits = v_feat_normed @ verb_text_features_norm.t()
        obj_logits = o_feat_normed @ obj_text_features_norm.t()


        verb_logits = verb_logits * 0.5 + 0.5
        obj_logits = obj_logits * 0.5 + 0.5


        # ===condition learning===
        b = video_features.shape[0]
        c = verb_text_features.shape[-1]
        n_v = verb_logits.shape[-1]
        n_o = obj_logits.shape[-1]

        # visual features
        o_feat_c = self.c2c_OE2(video_features.mean(dim=-1))
        v_feat_c_t = self.c2c_VE2(video_features)
        if self.use_gse:
            v_feat_c_t, o_feat_c = self.c2c_gse_cond(v_feat_c_t, o_feat_c)
        v_feat_c = v_feat_c_t.mean(dim=-1)

        p_v_con_o, p_o_con_v = self.condition_module(
            v_feat_c,
            o_feat_c,
            verb_text_features,
            obj_text_features,
            n_o,
            b,
            c,
            n_v,
        )
        p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)  # b,nv,no
        p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)  # b,nv,no

        global_pair_scores = p_pair_o + p_pair_v
        temporal_proposal_scores = None
        temporal_corrected_scores = None
        temporal_relation_diagnostics = None
        temporal_feedback_scale = None
        temporal_feedback_strength = None
        temporal_component_feedback_scores = None
        temporal_verb_unit_evidence = None
        temporal_object_unit_evidence = None
        temporal_shaper_scores = None
        if temporal_video_embedding is not None:
            if self.natural_pair_text_features.numel() == 0:
                raise RuntimeError(
                    "Natural pair text bank is empty. Call "
                    "prepare_natural_pair_text_bank() before forwarding."
                )
            (
                temporal_corrected_scores,
                temporal_proposal_scores,
                temporal_relation_diagnostics,
                temporal_feedback_scale,
                temporal_feedback_strength,
                temporal_component_feedback_scores,
                temporal_verb_unit_evidence,
                temporal_object_unit_evidence,
            ) = self.c2c_natural_pair_probabilistic_fusion(
                global_pair_scores,
                temporal_video_embedding,
                self.natural_pair_text_features,
            )
            if self.training and self.temporal_relation_shaper_enabled:
                temporal_shaper_scores = (
                    self._temporal_relation_shaper_scores(
                        video_features,
                        verb_text_features,
                        obj_text_features,
                        obj_logits,
                        global_pair_scores,
                    )
                )

        if self.training:
            outputs = (
                verb_logits,
                obj_logits,
                p_pair_v,
                p_pair_o,
                video_features,
                o_feat,
                v_feat,
                p_v_con_o,
                p_o_con_v,
            )
            if temporal_relation_diagnostics is not None:
                outputs = outputs + (
                    global_pair_scores.detach(),
                    temporal_proposal_scores,
                    temporal_corrected_scores,
                    temporal_gate_weights,
                    temporal_relation_diagnostics,
                    temporal_feedback_scale,
                    temporal_feedback_strength,
                    temporal_component_feedback_scores,
                    temporal_shaper_scores,
                )
            return outputs
        else:
            verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
            score_source = (
                temporal_corrected_scores
                if temporal_corrected_scores is not None
                else global_pair_scores
            )
            com_logits = score_source[:, verb_idx, obj_idx]
            if return_branch_scores and temporal_proposal_scores is not None:
                global_logits = global_pair_scores[:, verb_idx, obj_idx]
                proposal_logits = temporal_proposal_scores[:, verb_idx, obj_idx]
                verb_evidence_logits = temporal_verb_unit_evidence[
                    :, verb_idx, obj_idx
                ]
                object_evidence_logits = temporal_object_unit_evidence[
                    :, verb_idx, obj_idx
                ]
                return (
                    com_logits,
                    global_logits,
                    proposal_logits,
                    temporal_gate_weights,
                    temporal_relation_diagnostics,
                    temporal_feedback_scale,
                    temporal_feedback_strength,
                    verb_evidence_logits,
                    object_evidence_logits,
                )
            return com_logits

    def condition_module(
            self,
            v_feat_c,
            o_feat_c,
            v_emb,
            o_emb,
            n_o,
            b,
            c,
            n_v,
            apply_mmn=True,
            return_features=False):
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb_normed = F.normalize(o_emb, dim=1)

        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1,
                                                                                                                  c * 2))  # b,1,c+1,n,c -->b,n,c*2--->b,no,c
        f_v_e_o_norm = F.normalize(f_v_e_o, dim=-1)
        f_v_e_o_norm = f_v_e_o_norm.view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1,
                                                                                                                  c * 2))  # b,1,c+1,n,c -->b,n,c*2--->b,nv,c
        f_o_e_v_norm = F.normalize(f_o_e_v, dim=-1)
        f_o_e_v_norm = f_o_e_v_norm.view(b, n_v, c)

        if self.use_mmn_spatial_refiner and apply_mmn:
            f_v_e_o_norm, f_o_e_v_norm = self.c2c_mmn_spatial_refiner(
                f_v_e_o_norm,
                f_o_e_v_norm,
            )
            f_v_e_o_norm = F.normalize(f_v_e_o_norm, dim=-1)
            f_o_e_v_norm = F.normalize(f_o_e_v_norm, dim=-1)

        p_v_con_o = torch.einsum('bnc,mc->bnm', f_v_e_o_norm, v_emb_normed) * 0.5 + 0.5  # b,no,nv
        p_v_con_o = p_v_con_o.permute(0, 2, 1)  # b,nv,no
        p_o_con_v = torch.einsum('bnc,mc->bnm', f_o_e_v_norm, o_emb_normed) * 0.5 + 0.5  # b,nv,no
        if return_features:
            return (
                p_v_con_o,
                p_o_con_v,
                f_v_e_o_norm,
                f_o_e_v_norm,
            )
        return p_v_con_o, p_o_con_v


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def build_model(train_dataset,cfg):
    cfg = cfg
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg, train_dataset, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    # model.logit_scale
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                # if 'positional_embedding' in name:
                #     param.requires_grad_(True)
                #     print(f'{name}: {param.requires_grad}')
                if cfg.learn_input_method == 'coop':
                    if 'prompt_vectors' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'csp':
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'spm':
                    if 'prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                else:
                    raise NotImplementedError
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
                print(f'{name}: {param.requires_grad}')
        elif 'c2c' in name:
            param.requires_grad = True
            print(f'{name}: {param.requires_grad}')
    return model
