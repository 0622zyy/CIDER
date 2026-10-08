from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CED(nn.Module):
    """
    Spurious Co-occurrence Evidence Routing，CED。

    根据 Tail 标签注意力与高频共现 Head 标签注意力，
    将空间证据分解为：

    - Tail-private evidence；
    - Head/confounder evidence。

    Tail 私有证据形成新的类别专属增量 logit。
    """

    def __init__(
        self,
        num_classes: int,
        tail_indices: Tensor,
        confounder_indices: Tensor,
        confounder_weights: Tensor,
        tail_positive_weights: Tensor,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        confounder_strength: float,
        private_residual_floor: float,
        maximum_logit_correction: float,
        initial_gate_logit: float,
        negative_loss_weight: float,
        ranking_margin: float,
        evidence_margin: float,
        hard_negative_count: int,
        tail_bce_weight: float,
        ranking_weight: float,
        evidence_margin_weight: float,
        context_suppression_weight: float,
        attention_overlap_weight: float,
        feature_orthogonality_weight: float,
        correction_regularization_weight: float,
        routing_mode: str = "legacy",
        shared_evidence_strength: float = 0.5,
        route_prior_scale: float = 0.5,
        route_overlap_scale: float = 4.0,
        route_confidence_scale: float = 1.0,
        route_temperature: float = 0.5,
        route_overlap_center: float = 0.5,
        null_route_initial_logit: float = 1.0,
        route_supervision_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.num_tail_classes = int(
            tail_indices.numel()
        )

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.confounder_strength = float(
            confounder_strength
        )

        self.private_residual_floor = float(
            private_residual_floor
        )

        self.maximum_logit_correction = float(
            maximum_logit_correction
        )

        self.negative_loss_weight = float(
            negative_loss_weight
        )

        self.ranking_margin = float(
            ranking_margin
        )

        self.evidence_margin = float(
            evidence_margin
        )

        self.hard_negative_count = int(
            hard_negative_count
        )

        self.tail_bce_weight = float(
            tail_bce_weight
        )

        self.ranking_weight = float(
            ranking_weight
        )

        self.evidence_margin_weight = float(
            evidence_margin_weight
        )

        self.context_suppression_weight = float(
            context_suppression_weight
        )

        self.attention_overlap_weight = float(
            attention_overlap_weight
        )

        self.feature_orthogonality_weight = float(
            feature_orthogonality_weight
        )

        self.correction_regularization_weight = float(
            correction_regularization_weight
        )
        self.routing_mode = str(routing_mode)
        self.shared_evidence_strength = float(shared_evidence_strength)
        self.route_temperature = max(float(route_temperature), 1e-3)
        self.route_overlap_center = float(route_overlap_center)
        self.route_supervision_weight = float(route_supervision_weight)

        self.register_buffer(
            "tail_indices",
            tail_indices.clone().long(),
        )

        self.register_buffer(
            "confounder_indices",
            confounder_indices.clone().long(),
        )

        self.register_buffer(
            "confounder_weights",
            confounder_weights.clone().float(),
        )

        self.register_buffer(
            "tail_positive_weights",
            tail_positive_weights.clone().float(),
        )

        self.private_projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(
                self.feature_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
        )

        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(
                self.feature_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
        )

        self.class_embedding = nn.Embedding(
            self.num_tail_classes,
            self.hidden_dim,
        )

        self.classifier_weight = nn.Parameter(
            torch.empty(
                self.num_tail_classes,
                self.hidden_dim,
            )
        )

        self.classifier_bias = nn.Parameter(
            torch.zeros(
                self.num_tail_classes
            )
        )

        self.gate_logit = nn.Parameter(
            torch.full(
                (self.num_tail_classes,),
                fill_value=float(
                    initial_gate_logit
                ),
            )
        )

        if self.routing_mode == "adaptive_soft":
            # Softplus keeps every evidence contribution monotonic and makes
            # the learned routing coefficients directly interpretable.
            def inverse_softplus(value: float) -> float:
                value = max(float(value), 1e-4)
                return float(torch.log(torch.expm1(torch.tensor(value))).item())

            self.route_prior_raw = nn.Parameter(torch.tensor(
                inverse_softplus(route_prior_scale)
            ))
            self.route_overlap_raw = nn.Parameter(torch.tensor(
                inverse_softplus(route_overlap_scale)
            ))
            self.route_confidence_raw = nn.Parameter(torch.tensor(
                inverse_softplus(route_confidence_scale)
            ))
            self.null_route_logit = nn.Parameter(torch.full(
                (self.num_tail_classes,), float(null_route_initial_logit)
            ))

        nn.init.normal_(
            self.classifier_weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.class_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def _build_attention_components(
        self,
        attention: Tensor,
        baseline_logits: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        attention = attention.float()

        attention = (
            attention
            / attention.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        device = attention.device

        tail_indices = self.tail_indices.to(
            device
        )

        tail_attention = attention[
            :,
            tail_indices,
            :,
        ]

        confounder_indices = (
            self.confounder_indices.to(
                device
            )
        )

        valid = confounder_indices >= 0

        safe_indices = (
            confounder_indices.clamp_min(0)
        )

        selected_confounder_attention = (
            attention[
                :,
                safe_indices,
                :,
            ]
        )

        confounder_weights = self.confounder_weights.to(device) * valid.float()
        route_details: Dict[str, Tensor] = {}
        if self.routing_mode == "adaptive_soft":
            if baseline_logits is None:
                raise ValueError("adaptive_soft routing requires baseline_logits")

            candidate_overlap = torch.sqrt(
                tail_attention.unsqueeze(2).clamp_min(1e-8)
                * selected_confounder_attention.clamp_min(1e-8)
            ).sum(dim=-1)
            candidate_confidence = torch.sigmoid(
                baseline_logits.float()[:, safe_indices]
            ).detach()
            candidate_count = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            prior_term = torch.log(
                confounder_weights.clamp_min(1e-8) * candidate_count
            )
            route_logits = (
                F.softplus(self.route_prior_raw) * prior_term.unsqueeze(0)
                + F.softplus(self.route_overlap_raw)
                * (candidate_overlap - self.route_overlap_center)
                + F.softplus(self.route_confidence_raw)
                * torch.logit(candidate_confidence.clamp(1e-4, 1.0 - 1e-4))
            ) / self.route_temperature
            route_logits = route_logits.masked_fill(~valid.unsqueeze(0), -1e4)
            null_logits = self.null_route_logit.view(1, -1, 1).expand(
                attention.shape[0], -1, -1
            ) / self.route_temperature
            all_route_weights = torch.softmax(
                torch.cat((route_logits, null_logits), dim=2), dim=2
            )
            image_weights = all_route_weights[:, :, :-1] * valid.unsqueeze(0)
            null_route_weight = all_route_weights[:, :, -1]
            route_gate = image_weights.sum(dim=2)
            conditional_weights = image_weights / route_gate.unsqueeze(-1).clamp_min(1e-8)
            confounder_attention = (
                selected_confounder_attention * conditional_weights.unsqueeze(-1)
            ).sum(dim=2)
            route_details = {
                "route_weights": image_weights,
                "null_route_weight": null_route_weight,
                "route_gate": route_gate,
                "candidate_overlap": candidate_overlap * valid.unsqueeze(0),
                "candidate_confidence": candidate_confidence * valid.unsqueeze(0),
            }
        elif (
            self.routing_mode in {"image_soft", "dynamic_hard"}
            and baseline_logits is not None
        ):
            confidence = torch.sigmoid(
                baseline_logits.float()[:, safe_indices]
            ).detach()
            image_weights = confounder_weights.unsqueeze(0) * confidence
            image_weights = image_weights / image_weights.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-6)
            confounder_attention = (
                selected_confounder_attention * image_weights.unsqueeze(-1)
            ).sum(dim=2)
        else:
            confounder_weights = confounder_weights / confounder_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            confounder_attention = (
                selected_confounder_attention
                * confounder_weights.unsqueeze(0).unsqueeze(-1)
            ).sum(dim=2)

        confounder_attention = confounder_attention / confounder_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        if self.routing_mode == "image_soft":
            shared_saliency = confounder_attention / confounder_attention.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            private_raw = tail_attention * (
                self.private_residual_floor
                + (1.0 - self.private_residual_floor)
                * (1.0 - shared_saliency.detach())
            )
        else:
            routed_confounder_attention = confounder_attention
            if self.routing_mode == "adaptive_soft":
                routed_confounder_attention = (
                    routed_confounder_attention
                    * route_details["route_gate"].unsqueeze(-1)
                )
            private_raw = F.relu(
                tail_attention
                - self.confounder_strength * routed_confounder_attention
            )
            private_raw = (
                private_raw + self.private_residual_floor * tail_attention
            )

        private_attention = (
            private_raw
            / private_raw.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        overlap = torch.sqrt(
            tail_attention.clamp_min(1e-8)
            * confounder_attention.clamp_min(
                1e-8
            )
        ).sum(dim=-1)
        if self.routing_mode == "adaptive_soft":
            overlap = overlap * route_details["route_gate"]

        result = {
            "tail_attention": tail_attention,
            "confounder_attention": (
                confounder_attention
            ),
            "private_attention": (
                private_attention
            ),
            "attention_overlap": overlap,
        }
        result.update(route_details)
        return result

    def _evidence_logits(
        self,
        feature_map: Tensor,
        private_attention: Tensor,
        confounder_attention: Tensor,
        route_gate: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        spatial_feature = (
            feature_map.float()
            .flatten(2)
            .transpose(1, 2)
        )

        private_feature = torch.einsum(
            "bts,bsd->btd",
            private_attention,
            spatial_feature,
        )

        context_feature = torch.einsum(
            "bts,bsd->btd",
            confounder_attention,
            spatial_feature,
        )

        private_hidden = (
            self.private_projector(
                private_feature
            )
        )

        context_hidden = (
            self.context_projector(
                context_feature
            )
        )

        if route_gate is not None:
            context_hidden = context_hidden * route_gate.unsqueeze(-1)

        class_indices = torch.arange(
            self.num_tail_classes,
            device=feature_map.device,
        )

        class_embedding = (
            self.class_embedding(
                class_indices
            ).unsqueeze(0)
        )

        if self.routing_mode == "image_soft":
            private_hidden = torch.tanh(private_hidden + class_embedding)
        else:
            private_hidden = torch.tanh(
                private_hidden - context_hidden + class_embedding
            )

        context_hidden_for_logit = torch.tanh(
            context_hidden
            + class_embedding
        )

        private_logits = torch.einsum(
            "bth,th->bt",
            private_hidden,
            self.classifier_weight,
        )

        private_logits = (
            private_logits
            + self.classifier_bias
            .unsqueeze(0)
        )

        context_logits = torch.einsum(
            "bth,th->bt",
            context_hidden_for_logit,
            self.classifier_weight,
        )

        context_logits = (
            context_logits
            + self.classifier_bias
            .unsqueeze(0)
        )
        if route_gate is not None:
            context_logits = context_logits * route_gate

        return {
            "private_feature": private_feature,
            "context_feature": context_feature,
            "private_logits": private_logits,
            "context_logits": context_logits,
        }

    def _hard_ranking_loss(
        self,
        scores: Tensor,
        targets: Tensor,
    ) -> Tensor:
        losses = []

        for class_index in range(
            self.num_tail_classes
        ):
            positive_scores = scores[
                targets[:, class_index] > 0.5,
                class_index,
            ]

            negative_scores = scores[
                targets[:, class_index] <= 0.5,
                class_index,
            ]

            if (
                positive_scores.numel() == 0
                or negative_scores.numel() == 0
            ):
                continue

            hard_count = min(
                self.hard_negative_count,
                int(
                    negative_scores.numel()
                ),
            )

            hard_negatives = torch.topk(
                negative_scores,
                k=hard_count,
                largest=True,
                sorted=False,
            ).values

            difference = (
                positive_scores.unsqueeze(1)
                - hard_negatives.unsqueeze(0)
            )

            losses.append(
                F.softplus(
                    self.ranking_margin
                    - difference
                ).mean()
            )

        if not losses:
            return scores.sum() * 0.0

        return torch.stack(losses).mean()

    def forward(
        self,
        feature_map: Tensor,
        attention: Tensor,
        baseline_logits: Tensor,
        targets: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        components = (
            self._build_attention_components(
                attention,
                baseline_logits,
            )
        )

        evidence = self._evidence_logits(
            feature_map=feature_map,
            private_attention=components[
                "private_attention"
            ],
            confounder_attention=components[
                "confounder_attention"
            ],
            route_gate=components.get("route_gate"),
        )

        gate = torch.sigmoid(
            self.gate_logit
        ).view(
            1,
            self.num_tail_classes,
        )

        if self.routing_mode == "image_soft":
            raw_correction = gate * (
                evidence["private_logits"]
                - self.shared_evidence_strength * evidence["context_logits"]
            )
        elif self.routing_mode == "adaptive_soft":
            raw_correction = (
                gate * evidence["private_logits"] * components["route_gate"]
            )
        else:
            raw_correction = gate * evidence["private_logits"]

        correction = (
            self.maximum_logit_correction
            * torch.tanh(
                raw_correction
                / self.maximum_logit_correction
            )
        )

        corrected_logits = (
            baseline_logits.clone()
        )

        tail_indices = self.tail_indices.to(
            baseline_logits.device
        )

        corrected_tail_logits = (
            baseline_logits[
                :,
                tail_indices,
            ]
            + correction
        )

        corrected_logits[
            :,
            tail_indices,
        ] = corrected_tail_logits

        zero = corrected_logits.sum() * 0.0

        result = {
            "logits": corrected_logits,
            "tail_logits": corrected_tail_logits,
            "correction": correction,
            # Stable paper-facing aliases: shared evidence is the estimated
            # co-occurrence/confounder evidence and residual is its correction.
            "residual": correction,
            "private_logits": (
                evidence["private_logits"]
            ),
            "context_logits": (
                evidence["context_logits"]
            ),
            "private_feature": (
                evidence["private_feature"]
            ),
            "context_feature": (
                evidence["context_feature"]
            ),
            "shared_feature": (
                evidence["context_feature"]
            ),
            "private_attention": (
                components["private_attention"]
            ),
            "confounder_attention": (
                components[
                    "confounder_attention"
                ]
            ),
            "shared_attention": (
                components[
                    "confounder_attention"
                ]
            ),
            "attention_overlap": (
                components[
                    "attention_overlap"
                ]
            ),
            "gate": gate,
            "raw_auxiliary_loss": zero,
            "tail_bce_loss": zero,
            "ranking_loss": zero,
            "evidence_margin_loss": zero,
            "context_suppression_loss": zero,
            "attention_overlap_loss": zero,
            "feature_orthogonality_loss": zero,
            "correction_regularization": zero,
            "route_supervision_loss": zero,
        }
        for key in (
            "route_weights",
            "null_route_weight",
            "route_gate",
            "candidate_overlap",
            "candidate_confidence",
        ):
            if key in components:
                result[key] = components[key]

        if targets is None:
            return result

        tail_targets = targets[
            :,
            tail_indices,
        ].float()

        positive_mask = (
            tail_targets > 0.5
        )

        negative_mask = (
            ~positive_mask
        )

        route_supervision_loss = zero
        if self.routing_mode == "adaptive_soft":
            candidates = self.confounder_indices.to(targets.device)
            valid_candidates = candidates >= 0
            candidate_targets = targets[:, candidates.clamp_min(0)].float()
            head_present = (
                (candidate_targets > 0.5) & valid_candidates.unsqueeze(0)
            ).any(dim=2)
            valid_rows = valid_candidates.any(dim=1).unsqueeze(0).expand_as(head_present)
            route_probability = components["route_gate"].clamp(1e-6, 1.0 - 1e-6)
            positive_routes = head_present & valid_rows
            negative_routes = ~head_present & valid_rows
            route_parts = []
            if positive_routes.any():
                route_parts.append(-torch.log(route_probability[positive_routes]).mean())
            if negative_routes.any():
                route_parts.append(
                    -torch.log1p(-route_probability[negative_routes]).mean()
                )
            if route_parts:
                route_supervision_loss = torch.stack(route_parts).mean()

        positive_loss = (
            F.softplus(
                -corrected_tail_logits
            )
            * self.tail_positive_weights
            .to(
                corrected_tail_logits.device
            )
            .view(1, -1)
        )

        negative_loss = (
            F.softplus(
                corrected_tail_logits
            )
            * self.negative_loss_weight
        )

        tail_bce_loss = (
            (
                positive_loss
                * positive_mask.float()
            )
            + (
                negative_loss
                * negative_mask.float()
            )
        ).mean()

        ranking_loss = (
            self._hard_ranking_loss(
                scores=corrected_tail_logits,
                targets=tail_targets,
            )
        )

        if positive_mask.any():
            evidence_margin_loss = (
                F.softplus(
                    self.evidence_margin
                    + evidence[
                        "context_logits"
                    ][positive_mask]
                    - evidence[
                        "private_logits"
                    ][positive_mask]
                ).mean()
            )

            context_suppression_loss = (
                F.softplus(
                    evidence[
                        "context_logits"
                    ][positive_mask]
                ).mean()
            )

            attention_overlap_loss = (
                components[
                    "attention_overlap"
                ][positive_mask].mean()
            )

            cosine = F.cosine_similarity(
                evidence["private_feature"],
                evidence["context_feature"],
                dim=-1,
            ).abs()

            feature_orthogonality_loss = (
                cosine[
                    positive_mask
                ].mean()
            )
        else:
            evidence_margin_loss = zero
            context_suppression_loss = zero
            attention_overlap_loss = zero
            feature_orthogonality_loss = zero

        correction_regularization = (
            correction.square().mean()
        )

        raw_auxiliary_loss = (
            self.tail_bce_weight
            * tail_bce_loss
            + self.ranking_weight
            * ranking_loss
            + self.evidence_margin_weight
            * evidence_margin_loss
            + self.context_suppression_weight
            * context_suppression_loss
            + self.attention_overlap_weight
            * attention_overlap_loss
            + self.feature_orthogonality_weight
            * feature_orthogonality_loss
            + self.correction_regularization_weight
            * correction_regularization
            + self.route_supervision_weight * route_supervision_loss
        )

        result.update(
            {
                "raw_auxiliary_loss": (
                    raw_auxiliary_loss
                ),
                "tail_bce_loss": (
                    tail_bce_loss
                ),
                "ranking_loss": ranking_loss,
                "evidence_margin_loss": (
                    evidence_margin_loss
                ),
                "context_suppression_loss": (
                    context_suppression_loss
                ),
                "attention_overlap_loss": (
                    attention_overlap_loss
                ),
                "feature_orthogonality_loss": (
                    feature_orthogonality_loss
                ),
                "correction_regularization": (
                    correction_regularization
                ),
                "route_supervision_loss": route_supervision_loss,
            }
        )

        return result
