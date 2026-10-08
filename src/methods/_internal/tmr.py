from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _spherical_kmeans(
    features: Tensor,
    clusters: int,
    iterations: int,
    seed: int,
) -> Tensor:
    """Deterministic spherical k-means used only on official train features."""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [N,D] with N > 0")
    unit = F.normalize(features.float(), dim=1, eps=1e-6)
    count = min(int(clusters), int(unit.shape[0]))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(unit.shape[0], generator=generator, device="cpu")[:count]
    centers = unit.index_select(0, order.to(unit.device)).clone()

    for _ in range(max(int(iterations), 1)):
        assignment = (unit @ centers.transpose(0, 1)).argmax(dim=1)
        updated = []
        for cluster_index in range(count):
            selected = unit[assignment == cluster_index]
            if selected.numel() == 0:
                updated.append(centers[cluster_index])
            else:
                updated.append(F.normalize(selected.mean(dim=0), dim=0, eps=1e-6))
        next_centers = torch.stack(updated)
        if torch.allclose(next_centers, centers, atol=1e-5, rtol=0.0):
            centers = next_centers
            break
        centers = next_centers
    return centers


@torch.no_grad()
def build_tail_memories(
    private_features: Tensor,
    tail_targets: Tensor,
    base_tail_logits: Tensor,
    positive_prototypes: int = 3,
    negative_prototypes: int = 3,
    hard_negative_pool: int = 128,
    iterations: int = 20,
    seed: int = 42,
) -> Dict[str, Tensor]:
    """Create per-tail-class positive and hard-negative prototype memories."""

    if private_features.ndim != 3:
        raise ValueError("private_features must have shape [N,T,D]")
    if tail_targets.shape != private_features.shape[:2]:
        raise ValueError("tail_targets must have shape [N,T]")
    if base_tail_logits.shape != tail_targets.shape:
        raise ValueError("base_tail_logits must have shape [N,T]")

    samples, tail_classes, feature_dim = private_features.shape
    positive_memory = private_features.new_zeros(
        tail_classes, int(positive_prototypes), feature_dim, dtype=torch.float32
    )
    negative_memory = private_features.new_zeros(
        tail_classes, int(negative_prototypes), feature_dim, dtype=torch.float32
    )
    positive_valid = torch.zeros(
        tail_classes, int(positive_prototypes), dtype=torch.bool,
        device=private_features.device,
    )
    negative_valid = torch.zeros(
        tail_classes, int(negative_prototypes), dtype=torch.bool,
        device=private_features.device,
    )

    for class_index in range(tail_classes):
        positives = private_features[tail_targets[:, class_index] > 0.5, class_index]
        negative_indices = (tail_targets[:, class_index] <= 0.5).nonzero(
            as_tuple=False
        ).flatten()
        if positives.shape[0] == 0 or negative_indices.shape[0] == 0:
            raise RuntimeError(f"tail class {class_index} has no positive or negative memory")
        hard_count = min(int(hard_negative_pool), int(negative_indices.shape[0]))
        hard_order = base_tail_logits[negative_indices, class_index].topk(
            hard_count, largest=True, sorted=False
        ).indices
        hard_negatives = private_features[
            negative_indices.index_select(0, hard_order), class_index
        ]

        positive_centers = _spherical_kmeans(
            positives, positive_prototypes, iterations, seed + 101 * class_index
        )
        negative_centers = _spherical_kmeans(
            hard_negatives, negative_prototypes, iterations, seed + 103 * class_index
        )
        positive_memory[class_index, : positive_centers.shape[0]] = positive_centers
        negative_memory[class_index, : negative_centers.shape[0]] = negative_centers
        positive_valid[class_index, : positive_centers.shape[0]] = True
        negative_valid[class_index, : negative_centers.shape[0]] = True

    return {
        "positive_memory": positive_memory,
        "negative_memory": negative_memory,
        "positive_valid": positive_valid,
        "negative_valid": negative_valid,
        "num_samples": torch.tensor(samples),
    }


@torch.no_grad()
def build_tail_dependency_graph(
    targets: Tensor,
    tail_indices: Tensor,
    smoothing: float = 1.0,
    topk: int = 10,
    shrinkage: float = 10.0,
) -> Dict[str, Tensor]:
    """Estimate signed train-only label dependencies for every Tail class."""
    targets = targets.float()
    samples, classes = targets.shape
    graph = targets.new_zeros((tail_indices.numel(), classes))
    alpha = float(smoothing)
    for row, class_index in enumerate(tail_indices.tolist()):
        positive = targets[:, class_index] > 0.5
        negative = ~positive
        positive_count = positive.sum().float().clamp_min(1.0)
        negative_count = negative.sum().float().clamp_min(1.0)
        p_positive = (targets[positive].sum(dim=0) + alpha) / (
            positive_count + 2.0 * alpha
        )
        p_negative = (targets[negative].sum(dim=0) + alpha) / (
            negative_count + 2.0 * alpha
        )
        log_odds_positive = torch.logit(p_positive.clamp(1e-4, 1.0 - 1e-4))
        log_odds_negative = torch.logit(p_negative.clamp(1e-4, 1.0 - 1e-4))
        score = log_odds_positive - log_odds_negative
        score[class_index] = 0.0
        score *= positive_count / (positive_count + float(shrinkage))
        keep = min(max(int(topk), 1), classes - 1)
        selected = score.abs().topk(keep, largest=True, sorted=False).indices
        graph[row, selected] = score[selected]
    graph = F.normalize(graph, dim=1, eps=1e-6)
    return {
        "label_graph": graph,
        "class_prior": targets.mean(dim=0),
        "num_samples": torch.tensor(samples),
    }


class TMR(nn.Module):
    """Legacy implementation used internally by CEDR reliability refinement."""

    def __init__(
        self,
        num_classes: int,
        tail_indices: Tensor,
        positive_memory: Tensor,
        negative_memory: Tensor,
        positive_valid: Tensor,
        negative_valid: Tensor,
        maximum_residual: float = 3.0,
        uncertainty_temperature: float = 2.0,
        ranking_margin: float = 1.0,
        ranking_weight: float = 0.25,
        residual_regularization_weight: float = 0.002,
        adapter_deviation: float = 0.25,
        adapter_regularization_weight: float = 0.0005,
        adapter_initial_temperature: float = 5.0,
        adapter_classification_weight: float = 1.0,
        adapter_ranking_weight: float = 0.25,
        label_graph: Tensor | None = None,
        class_prior: Tensor | None = None,
        dependency_deviation: float = 0.5,
        dependency_initial_temperature: float = 3.0,
        dependency_initial_scale: float = 0.0,
        dependency_classification_weight: float = 1.0,
        dependency_ranking_weight: float = 0.25,
        dependency_regularization_weight: float = 0.0005,
        dependency_private_gate_temperature: float = 3.0,
        shared_positive_memory: Tensor | None = None,
        shared_negative_memory: Tensor | None = None,
        shared_positive_valid: Tensor | None = None,
        shared_negative_valid: Tensor | None = None,
        shared_excess_strength: float = 1.0,
        causal_initial_temperature: float = 5.0,
        causal_classification_weight: float = 1.0,
        causal_ranking_weight: float = 0.5,
        attention_gate_strength: float = 1.0,
        verification_mean: Tensor | None = None,
        verification_std: Tensor | None = None,
        verification_initial_temperature: float = 3.0,
        verification_classification_weight: float = 1.0,
        verification_ranking_weight: float = 0.5,
        positive_gate_mode: str = "legacy",
        prototype_margin: float = 0.0,
        private_shared_gap_margin: float = 0.0,
        private_confidence_threshold: float = 0.2,
        positive_gate_temperature: float = 4.0,
        uncertainty_dead_zone: float = 0.0,
        uncertainty_gate_temperature: float = 4.0,
        maximum_positive_residual: float | None = None,
        maximum_negative_residual: float | None = None,
        confounder_indices: Tensor | None = None,
        common_head_ranking_weight: float = 0.0,
        common_head_ranking_margin: float = 1.0,
        dependency_enabled: bool = True,
    ) -> None:
        super().__init__()
        if positive_memory.ndim != 3 or negative_memory.ndim != 3:
            raise ValueError("memory tensors must have shape [T,K,D]")
        tail_classes = int(tail_indices.numel())
        if positive_memory.shape[0] != tail_classes:
            raise ValueError("positive memory class count does not match tail_indices")
        if negative_memory.shape[0] != tail_classes:
            raise ValueError("negative memory class count does not match tail_indices")
        if positive_valid.shape != positive_memory.shape[:2]:
            raise ValueError("positive_valid has an invalid shape")
        if negative_valid.shape != negative_memory.shape[:2]:
            raise ValueError("negative_valid has an invalid shape")
        if maximum_residual <= 0 or uncertainty_temperature <= 0:
            raise ValueError("residual and uncertainty scales must be positive")

        self.num_classes = int(num_classes)
        self.num_tail_classes = tail_classes
        self.maximum_residual = float(maximum_residual)
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.ranking_margin = float(ranking_margin)
        self.ranking_weight = float(ranking_weight)
        self.residual_regularization_weight = float(
            residual_regularization_weight
        )
        self.adapter_deviation = float(adapter_deviation)
        self.adapter_regularization_weight = float(adapter_regularization_weight)
        self.adapter_classification_weight = float(adapter_classification_weight)
        self.adapter_ranking_weight = float(adapter_ranking_weight)
        self.dependency_deviation = float(dependency_deviation)
        self.dependency_classification_weight = float(
            dependency_classification_weight
        )
        self.dependency_ranking_weight = float(dependency_ranking_weight)
        self.dependency_regularization_weight = float(
            dependency_regularization_weight
        )
        self.dependency_private_gate_temperature = float(
            dependency_private_gate_temperature
        )
        self.shared_excess_strength = float(shared_excess_strength)
        self.causal_classification_weight = float(
            causal_classification_weight
        )
        self.causal_ranking_weight = float(causal_ranking_weight)
        self.attention_gate_strength = float(attention_gate_strength)
        self.verification_classification_weight = float(
            verification_classification_weight
        )
        self.verification_ranking_weight = float(verification_ranking_weight)
        self.positive_gate_mode = str(positive_gate_mode)
        if self.positive_gate_mode not in {"legacy", "triple"}:
            raise ValueError("positive_gate_mode must be legacy or triple")
        self.prototype_margin = float(prototype_margin)
        self.private_shared_gap_margin = float(private_shared_gap_margin)
        self.private_confidence_threshold = float(private_confidence_threshold)
        self.positive_gate_temperature = max(float(positive_gate_temperature), 1e-3)
        self.uncertainty_dead_zone = float(uncertainty_dead_zone)
        self.uncertainty_gate_temperature = max(
            float(uncertainty_gate_temperature), 1e-3
        )
        self.maximum_positive_residual = float(
            maximum_residual
            if maximum_positive_residual is None
            else maximum_positive_residual
        )
        self.maximum_negative_residual = float(
            maximum_residual
            if maximum_negative_residual is None
            else maximum_negative_residual
        )
        if self.maximum_positive_residual <= 0 or self.maximum_negative_residual <= 0:
            raise ValueError("asymmetric residual bounds must be positive")
        self.common_head_ranking_weight = float(common_head_ranking_weight)
        self.common_head_ranking_margin = float(common_head_ranking_margin)
        self.dependency_enabled = bool(dependency_enabled)
        self.register_buffer("tail_indices", tail_indices.long().clone())
        if confounder_indices is None:
            confounder_indices = torch.full((tail_classes, 1), -1, dtype=torch.long)
        if confounder_indices.ndim != 2 or confounder_indices.shape[0] != tail_classes:
            raise ValueError("confounder_indices must have shape [T,K]")
        self.register_buffer(
            "confounder_indices", confounder_indices.long().clone(), persistent=False
        )
        self.register_buffer(
            "positive_memory", F.normalize(positive_memory.float(), dim=-1, eps=1e-6)
        )
        self.register_buffer(
            "negative_memory", F.normalize(negative_memory.float(), dim=-1, eps=1e-6)
        )
        self.register_buffer("positive_valid", positive_valid.bool().clone())
        self.register_buffer("negative_valid", negative_valid.bool().clone())
        if shared_positive_memory is None:
            shared_positive_memory = positive_memory
        if shared_negative_memory is None:
            shared_negative_memory = negative_memory
        if shared_positive_valid is None:
            shared_positive_valid = positive_valid
        if shared_negative_valid is None:
            shared_negative_valid = negative_valid
        if shared_positive_memory.shape != positive_memory.shape:
            raise ValueError("shared positive memory has an invalid shape")
        if shared_negative_memory.shape != negative_memory.shape:
            raise ValueError("shared negative memory has an invalid shape")
        self.register_buffer(
            "shared_positive_memory",
            F.normalize(shared_positive_memory.float(), dim=-1, eps=1e-6),
        )
        self.register_buffer(
            "shared_negative_memory",
            F.normalize(shared_negative_memory.float(), dim=-1, eps=1e-6),
        )
        self.register_buffer(
            "shared_positive_valid", shared_positive_valid.bool().clone()
        )
        self.register_buffer(
            "shared_negative_valid", shared_negative_valid.bool().clone()
        )

        positive_count = positive_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        negative_count = negative_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        positive_center = (
            self.positive_memory * positive_valid[..., None].float()
        ).sum(dim=1) / positive_count
        negative_center = (
            self.negative_memory * negative_valid[..., None].float()
        ).sum(dim=1) / negative_count
        self.register_buffer(
            "anchor_direction",
            F.normalize(positive_center - negative_center, dim=-1, eps=1e-6),
        )
        if label_graph is None:
            label_graph = torch.zeros(tail_classes, self.num_classes)
        if class_prior is None:
            class_prior = torch.zeros(self.num_classes)
        if tuple(label_graph.shape) != (tail_classes, self.num_classes):
            raise ValueError("label_graph must have shape [T,C]")
        if tuple(class_prior.shape) != (self.num_classes,):
            raise ValueError("class_prior must have shape [C]")
        self.register_buffer(
            "label_graph", F.normalize(label_graph.float(), dim=1, eps=1e-6)
        )
        self.register_buffer("class_prior", class_prior.float().clone())
        if verification_mean is None:
            verification_mean = torch.zeros(tail_classes)
        if verification_std is None:
            verification_std = torch.ones(tail_classes)
        if verification_mean.shape != (tail_classes,):
            raise ValueError("verification_mean must have shape [T]")
        if verification_std.shape != (tail_classes,):
            raise ValueError("verification_std must have shape [T]")
        self.register_buffer("verification_mean", verification_mean.float().clone())
        self.register_buffer(
            "verification_std", verification_std.float().clamp_min(1e-4).clone()
        )

        # Zero initialization makes M3 exactly equal to locked M2 before fitting.
        self.positive_scale = nn.Parameter(torch.zeros(tail_classes))
        self.negative_scale = nn.Parameter(torch.zeros(tail_classes))
        self.interaction_scale = nn.Parameter(torch.zeros(tail_classes))
        self.adapter_scale = nn.Parameter(torch.zeros(tail_classes))
        self.adapter_delta = nn.Parameter(torch.zeros_like(self.anchor_direction))
        initial_temperature = torch.full(
            (tail_classes,), float(adapter_initial_temperature)
        ).clamp_min(1e-3)
        self.adapter_log_temperature = nn.Parameter(
            torch.log(torch.expm1(initial_temperature))
        )
        self.adapter_bias = nn.Parameter(torch.zeros(tail_classes))
        self.dependency_scale = nn.Parameter(
            torch.full((tail_classes,), float(dependency_initial_scale))
        )
        self.dependency_delta = nn.Parameter(torch.zeros_like(self.label_graph))
        dependency_temperature = torch.full(
            (tail_classes,), float(dependency_initial_temperature)
        ).clamp_min(1e-3)
        self.dependency_log_temperature = nn.Parameter(
            torch.log(torch.expm1(dependency_temperature))
        )
        self.dependency_bias = nn.Parameter(torch.zeros(tail_classes))
        causal_temperature = torch.full(
            (tail_classes,), float(causal_initial_temperature)
        ).clamp_min(1e-3)
        self.causal_log_temperature = nn.Parameter(
            torch.log(torch.expm1(causal_temperature))
        )
        self.causal_bias = nn.Parameter(torch.zeros(tail_classes))
        self.causal_scale = nn.Parameter(torch.zeros(tail_classes))
        verification_temperature = torch.full(
            (tail_classes,), float(verification_initial_temperature)
        ).clamp_min(1e-3)
        self.verification_log_temperature = nn.Parameter(
            torch.log(torch.expm1(verification_temperature))
        )
        self.verification_bias = nn.Parameter(torch.zeros(tail_classes))
        self.verification_scale = nn.Parameter(torch.zeros(tail_classes))
        self.class_bias = nn.Parameter(torch.zeros(tail_classes))

    def _memory_scores(self, private_features: Tensor) -> tuple[Tensor, Tensor]:
        if private_features.ndim != 3:
            raise ValueError("private_features must have shape [B,T,D]")
        expected = (self.num_tail_classes, self.positive_memory.shape[-1])
        if tuple(private_features.shape[1:]) != expected:
            raise ValueError(f"private feature tail shape must be {expected}")
        unit = F.normalize(private_features.float(), dim=-1, eps=1e-6)
        positive_similarity = torch.einsum(
            "btd,tkd->btk", unit, self.positive_memory
        ).masked_fill(~self.positive_valid[None], -1e4)
        negative_similarity = torch.einsum(
            "btd,tkd->btk", unit, self.negative_memory
        ).masked_fill(~self.negative_valid[None], -1e4)
        return positive_similarity.max(dim=-1).values, negative_similarity.max(dim=-1).values

    def _shared_memory_scores(self, shared_features: Tensor) -> tuple[Tensor, Tensor]:
        if shared_features.ndim != 3:
            raise ValueError("shared_features must have shape [B,T,D]")
        expected = (self.num_tail_classes, self.shared_positive_memory.shape[-1])
        if tuple(shared_features.shape[1:]) != expected:
            raise ValueError(f"shared feature tail shape must be {expected}")
        unit = F.normalize(shared_features.float(), dim=-1, eps=1e-6)
        positive_similarity = torch.einsum(
            "btd,tkd->btk", unit, self.shared_positive_memory
        ).masked_fill(~self.shared_positive_valid[None], -1e4)
        negative_similarity = torch.einsum(
            "btd,tkd->btk", unit, self.shared_negative_memory
        ).masked_fill(~self.shared_negative_valid[None], -1e4)
        return (
            positive_similarity.max(dim=-1).values,
            negative_similarity.max(dim=-1).values,
        )

    def _hard_ranking_loss(self, scores: Tensor, targets: Tensor) -> Tensor:
        losses = []
        for class_index in range(self.num_tail_classes):
            positive = scores[targets[:, class_index] > 0.5, class_index]
            negative = scores[targets[:, class_index] <= 0.5, class_index]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            hard_count = min(16, int(negative.numel()))
            hard_negative = negative.topk(hard_count, largest=True).values
            losses.append(
                F.softplus(
                    self.ranking_margin
                    - positive[:, None]
                    + hard_negative[None, :]
                ).mean()
            )
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()

    def _common_head_ranking_loss(
        self,
        scores: Tensor,
        tail_targets: Tensor,
        full_targets: Tensor,
    ) -> Tensor:
        """Rank de-co-occurred Tail positives above Head-only hard negatives."""
        losses = []
        candidates = self.confounder_indices.to(full_targets.device)
        for class_index in range(self.num_tail_classes):
            valid = candidates[class_index] >= 0
            if not valid.any():
                continue
            head_present = full_targets[:, candidates[class_index, valid]].amax(dim=1) > 0.5
            tail_present = tail_targets[:, class_index] > 0.5
            positive = scores[tail_present & ~head_present, class_index]
            negative = scores[~tail_present & head_present, class_index]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            hard_count = min(16, int(negative.numel()))
            hard_negative = negative.topk(hard_count, largest=True).values
            losses.append(
                F.softplus(
                    self.common_head_ranking_margin
                    - positive[:, None]
                    + hard_negative[None, :]
                ).mean()
            )
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        base_logits: Tensor,
        private_features: Tensor,
        shared_features: Tensor | None = None,
        attention_separation: Tensor | None = None,
        verification_score: Tensor | None = None,
        targets: Tensor | None = None,
        positive_weights: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if base_logits.ndim != 2 or base_logits.shape[1] != self.num_classes:
            raise ValueError("base_logits must have shape [B,C]")
        positive_score, negative_score = self._memory_scores(private_features)
        contrast = positive_score - negative_score
        if shared_features is None:
            shared_features = private_features
        shared_positive_score, shared_negative_score = (
            self._shared_memory_scores(shared_features)
        )
        shared_contrast = shared_positive_score - shared_negative_score
        context_excess = F.relu(shared_contrast - contrast)
        causal_contrast = (
            contrast - self.shared_excess_strength * context_excess
        )
        causal_temperature = F.softplus(
            self.causal_log_temperature
        ).clamp_max(20.0)
        causal_logit = (
            causal_temperature[None] * causal_contrast
            + self.causal_bias[None]
        )
        if verification_score is None:
            verification_score = contrast.new_zeros(contrast.shape)
        if verification_score.shape != contrast.shape:
            raise ValueError("verification_score must have shape [B,T]")
        normalized_verification = (
            verification_score.float() - self.verification_mean[None]
        ) / self.verification_std[None]
        verification_temperature = F.softplus(
            self.verification_log_temperature
        ).clamp_max(20.0)
        verification_logit = (
            verification_temperature[None] * normalized_verification
            + self.verification_bias[None]
        )
        unit_private = F.normalize(private_features.float(), dim=-1, eps=1e-6)
        adapter_direction = F.normalize(
            self.anchor_direction
            + self.adapter_deviation * torch.tanh(self.adapter_delta),
            dim=-1,
            eps=1e-6,
        )
        adapter_score = torch.einsum("btd,td->bt", unit_private, adapter_direction)
        adapter_temperature = F.softplus(self.adapter_log_temperature).clamp_max(20.0)
        adapter_logit = (
            adapter_temperature[None] * adapter_score + self.adapter_bias[None]
        )
        dependency_direction = F.normalize(
            self.label_graph
            + self.dependency_deviation * torch.tanh(self.dependency_delta),
            dim=1,
            eps=1e-6,
        )
        centered_probability = torch.sigmoid(base_logits.float()) - self.class_prior[None]
        dependency_score = torch.einsum(
            "bc,tc->bt", centered_probability, dependency_direction
        )
        dependency_temperature = F.softplus(
            self.dependency_log_temperature
        ).clamp_max(20.0)
        dependency_logit = (
            dependency_temperature[None] * dependency_score
            + self.dependency_bias[None]
        )
        if attention_separation is None:
            attention_separation = contrast.new_full(contrast.shape, 0.5)
        if attention_separation.shape != contrast.shape:
            raise ValueError("attention_separation must have shape [B,T]")
        attention_signal = 2.0 * attention_separation.float() - 1.0
        dependency_private_gate = torch.sigmoid(
            self.dependency_private_gate_temperature
            * (
                contrast
                + self.attention_gate_strength * attention_signal
            )
        )
        # Positive relational support requires private object evidence; signed
        # negative evidence remains active to suppress likely false positives.
        gated_dependency_logit = (
            dependency_logit
            - F.relu(dependency_logit) * (1.0 - dependency_private_gate)
        )
        private_shared_gap = contrast - shared_contrast
        prototype_gate = torch.sigmoid(
            self.positive_gate_temperature * (contrast - self.prototype_margin)
        )
        private_shared_gate = torch.sigmoid(
            self.positive_gate_temperature
            * (private_shared_gap - self.private_shared_gap_margin)
        )
        private_confidence_gate = torch.sigmoid(
            self.positive_gate_temperature
            * (positive_score - self.private_confidence_threshold)
        )
        positive_evidence_gate = (
            prototype_gate * private_shared_gate * private_confidence_gate
        )
        tail_indices = self.tail_indices.to(base_logits.device)
        base_tail = base_logits[:, tail_indices].float()
        uncertainty = torch.exp(
            -base_tail.abs() / self.uncertainty_temperature
        )
        if self.uncertainty_dead_zone > 0:
            uncertainty_gate = torch.sigmoid(
                self.uncertainty_gate_temperature
                * (contrast.abs() - self.uncertainty_dead_zone)
            )
        else:
            uncertainty_gate = torch.ones_like(base_tail)
        raw_residual = (
            self.positive_scale[None] * positive_score
            - self.negative_scale[None] * negative_score
            + self.interaction_scale[None] * contrast * uncertainty
            + self.adapter_scale[None] * adapter_logit
            + float(self.dependency_enabled)
            * self.dependency_scale[None]
            * gated_dependency_logit
            + self.causal_scale[None] * causal_logit
            + self.verification_scale[None] * verification_logit * uncertainty
            + self.class_bias[None]
        )
        if self.positive_gate_mode == "triple":
            raw_residual = (
                raw_residual
                - F.relu(raw_residual) * (1.0 - positive_evidence_gate)
            )
        residual = (
            self.maximum_positive_residual
            * torch.tanh(F.relu(raw_residual) / self.maximum_positive_residual)
            - self.maximum_negative_residual
            * torch.tanh(F.relu(-raw_residual) / self.maximum_negative_residual)
        )
        residual = residual * uncertainty_gate
        tail_logits = base_tail + residual
        logits = base_logits.clone()
        logits[:, tail_indices] = tail_logits.to(logits.dtype)
        zero = logits.sum() * 0.0
        classification = zero
        ranking = zero
        common_head_ranking = zero
        residual_regularization = residual.square().mean()
        adapter_regularization = self.adapter_delta.square().mean()
        adapter_classification = zero
        adapter_ranking = zero
        dependency_classification = zero
        dependency_ranking = zero
        dependency_regularization = self.dependency_delta.square().mean()
        causal_classification = zero
        causal_ranking = zero
        verification_classification = zero
        verification_ranking = zero

        if targets is not None:
            tail_targets = targets[:, tail_indices].float()
            if positive_weights is None:
                class_weight = torch.ones(
                    self.num_tail_classes, device=tail_logits.device
                )
            else:
                if positive_weights.shape != (self.num_tail_classes,):
                    raise ValueError("positive_weights must have shape [T]")
                class_weight = positive_weights.to(tail_logits)
            classification = (
                tail_targets * class_weight[None] * F.softplus(-tail_logits)
                + (1.0 - tail_targets) * F.softplus(tail_logits)
            ).mean()
            ranking = self._hard_ranking_loss(tail_logits, tail_targets)
            common_head_ranking = self._common_head_ranking_loss(
                tail_logits, tail_targets, targets.float()
            )
            adapter_classification = (
                tail_targets * class_weight[None] * F.softplus(-adapter_logit)
                + (1.0 - tail_targets) * F.softplus(adapter_logit)
            ).mean()
            adapter_ranking = self._hard_ranking_loss(adapter_logit, tail_targets)
            dependency_classification = (
                tail_targets * class_weight[None] * F.softplus(-dependency_logit)
                + (1.0 - tail_targets) * F.softplus(dependency_logit)
            ).mean()
            dependency_ranking = self._hard_ranking_loss(
                dependency_logit, tail_targets
            )
            causal_classification = (
                tail_targets * class_weight[None] * F.softplus(-causal_logit)
                + (1.0 - tail_targets) * F.softplus(causal_logit)
            ).mean()
            causal_ranking = self._hard_ranking_loss(
                causal_logit, tail_targets
            )
            verification_classification = (
                tail_targets
                * class_weight[None]
                * F.softplus(-verification_logit)
                + (1.0 - tail_targets) * F.softplus(verification_logit)
            ).mean()
            verification_ranking = self._hard_ranking_loss(
                verification_logit, tail_targets
            )

        loss = (
            classification
            + self.ranking_weight * ranking
            + self.common_head_ranking_weight * common_head_ranking
            + self.residual_regularization_weight * residual_regularization
            + self.adapter_regularization_weight * adapter_regularization
            + self.adapter_classification_weight * adapter_classification
            + self.adapter_ranking_weight * adapter_ranking
            + self.dependency_classification_weight * dependency_classification
            + self.dependency_ranking_weight * dependency_ranking
            + self.dependency_regularization_weight * dependency_regularization
            + self.causal_classification_weight * causal_classification
            + self.causal_ranking_weight * causal_ranking
            + self.verification_classification_weight
            * verification_classification
            + self.verification_ranking_weight * verification_ranking
        )
        return {
            "logits": logits,
            "tail_logits": tail_logits,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "contrast_score": contrast,
            "shared_positive_score": shared_positive_score,
            "shared_negative_score": shared_negative_score,
            "shared_contrast_score": shared_contrast,
            "context_excess": context_excess,
            "causal_contrast": causal_contrast,
            "causal_logit": causal_logit,
            "verification_logit": verification_logit,
            "uncertainty": uncertainty,
            "uncertainty_gate": uncertainty_gate,
            "adapter_score": adapter_score,
            "adapter_logit": adapter_logit,
            "dependency_score": dependency_score,
            "dependency_logit": dependency_logit,
            "dependency_private_gate": dependency_private_gate,
            "prototype_gate": prototype_gate,
            "private_shared_gap": private_shared_gap,
            "private_shared_gate": private_shared_gate,
            "private_confidence_gate": private_confidence_gate,
            "positive_evidence_gate": positive_evidence_gate,
            "gated_dependency_logit": gated_dependency_logit,
            "residual": residual,
            "loss_classification": classification,
            "loss_ranking": ranking,
            "loss_common_head_ranking": common_head_ranking,
            "loss_residual_regularization": residual_regularization,
            "loss_adapter_regularization": adapter_regularization,
            "loss_adapter_classification": adapter_classification,
            "loss_adapter_ranking": adapter_ranking,
            "loss_dependency_classification": dependency_classification,
            "loss_dependency_ranking": dependency_ranking,
            "loss_dependency_regularization": dependency_regularization,
            "loss_causal_classification": causal_classification,
            "loss_causal_ranking": causal_ranking,
            "loss_verification_classification": verification_classification,
            "loss_verification_ranking": verification_ranking,
            "loss": loss,
        }


__all__ = [
    "TMR",
    "build_tail_dependency_graph",
    "build_tail_memories",
]
