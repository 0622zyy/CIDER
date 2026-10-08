from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LCI(nn.Module):
    """
    Tail-aware Label-Conditional Context Intervention，LCI。

    对每张多标签图像优先选择一个 Tail 正标签，通过该标签
    在 ML-Decoder 中的 cross-attention 定位标签核心区域。

    构造两种干预：

    1. Context swap
       保留目标标签核心特征，换入一张不包含该标签图像的上下文。

    2. Core removal
       保留原上下文，用该图像上下文均值替换标签核心区域。

    优化目标：

    - 换上下文后目标标签仍为正；
    - 换上下文前后目标 logit 稳定；
    - 移除核心区域后目标 logit 应明显下降。
    """

    def __init__(
        self,
        num_classes: int,
        class_split: Mapping[str, Sequence[int]],
        class_frequency: Tensor,
        core_topk_ratio: float,
        necessity_margin: float,
        swap_positive_weight: float,
        swap_invariance_weight: float,
        core_necessity_weight: float,
        auxiliary_to_classification_cap: float,
        head_sample_weight: float,
        middle_sample_weight: float,
        tail_sample_weight: float,
        mode: str = "legacy",
        soft_mask_temperature: float = 0.5,
        ranking_margin: float = 1.0,
        ranking_weight: float = 0.2,
        maximum_frequency_weight: float = 3.0,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.core_topk_ratio = float(core_topk_ratio)
        self.necessity_margin = float(necessity_margin)

        self.swap_positive_weight = float(
            swap_positive_weight
        )

        self.swap_invariance_weight = float(
            swap_invariance_weight
        )

        self.core_necessity_weight = float(
            core_necessity_weight
        )

        self.auxiliary_to_classification_cap = float(
            auxiliary_to_classification_cap
        )
        self.mode = str(mode)
        self.soft_mask_temperature = float(soft_mask_temperature)
        self.ranking_margin = float(ranking_margin)
        self.ranking_weight = float(ranking_weight)
        self.maximum_frequency_weight = float(maximum_frequency_weight)

        group_ids = torch.full(
            (self.num_classes,),
            fill_value=-1,
            dtype=torch.long,
        )

        group_weights = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
        )

        selection_priority = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
        )

        specifications = (
            (
                "head",
                0,
                float(head_sample_weight),
                1.0,
            ),
            (
                "middle",
                1,
                float(middle_sample_weight),
                2.0,
            ),
            (
                "tail",
                2,
                float(tail_sample_weight),
                3.0,
            ),
        )

        for (
            group_name,
            group_id,
            group_weight,
            priority,
        ) in specifications:
            for class_index in class_split[
                group_name
            ]:
                class_index = int(class_index)

                group_ids[class_index] = group_id
                group_weights[class_index] = (
                    group_weight
                )
                selection_priority[class_index] = (
                    priority
                )

        if torch.any(group_ids < 0):
            raise RuntimeError(
                "Head/Middle/Tail 没有完整覆盖类别"
            )

        frequency = class_frequency.float().clamp_min(
            1.0
        )

        inverse_frequency = (
            1.0 / torch.log1p(frequency)
        )

        inverse_frequency = (
            inverse_frequency
            / inverse_frequency.max()
        )

        self.register_buffer(
            "group_ids",
            group_ids,
        )

        self.register_buffer(
            "group_weights",
            group_weights,
        )

        self.register_buffer(
            "selection_priority",
            selection_priority,
        )

        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
        )

    def _select_positive_class(
        self,
        targets: Tensor,
    ) -> Tensor:
        positive_mask = targets > 0.5

        if not positive_mask.any(dim=1).all():
            raise RuntimeError(
                "LCI 遇到没有正标签的训练样本"
            )

        scores = (
            self.selection_priority
            + 0.25 * self.inverse_frequency
        ).view(
            1,
            self.num_classes,
        ).expand_as(targets)

        scores = torch.where(
            positive_mask,
            scores,
            torch.full_like(
                scores,
                -1e8,
            ),
        )

        return scores.argmax(dim=1)

    def _build_core_mask(
        self,
        selected_attention: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        batch_size, token_count = (
            selected_attention.shape
        )

        if token_count != height * width:
            raise RuntimeError(
                "Attention token 数与特征图不一致"
            )

        topk_count = max(
            1,
            min(
                token_count - 1,
                int(round(
                    token_count
                    * self.core_topk_ratio
                )),
            ),
        )

        topk_indices = torch.topk(
            selected_attention,
            k=topk_count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

        mask = torch.zeros(
            batch_size,
            token_count,
            dtype=selected_attention.dtype,
            device=selected_attention.device,
        )

        mask.scatter_(
            dim=1,
            index=topk_indices,
            value=1.0,
        )

        return mask.view(
            batch_size,
            1,
            height,
            width,
        ).detach()

    @staticmethod
    def _select_donors(
        targets: Tensor,
        selected_classes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = targets.shape[0]

        donor_indices = []
        valid_donors = []

        for sample_index in range(batch_size):
            selected_class = int(
                selected_classes[
                    sample_index
                ].item()
            )

            candidates = torch.nonzero(
                targets[:, selected_class] <= 0.5,
                as_tuple=False,
            ).flatten()

            candidates = candidates[
                candidates != sample_index
            ]

            if candidates.numel() == 0:
                # Use the anchor only as an indexing-safe placeholder.  The
                # corresponding validity flag is false, so this sample has
                # zero weight in every LCI auxiliary loss below.
                donor_indices.append(
                    sample_index
                )
                valid_donors.append(False)
                continue

            source_labels = targets[
                sample_index
            ].unsqueeze(0)

            candidate_labels = targets[
                candidates
            ]

            label_overlap = (
                source_labels * candidate_labels
            ).sum(dim=1)

            chosen = candidates[
                torch.argmin(label_overlap)
            ]

            donor_indices.append(
                int(chosen.item())
            )
            valid_donors.append(True)

        return (
            torch.tensor(
                donor_indices,
                dtype=torch.long,
                device=targets.device,
            ),
            torch.tensor(
                valid_donors,
                dtype=torch.bool,
                device=targets.device,
            ),
        )

    @staticmethod
    def _weighted_mean(
        values: Tensor,
        weights: Tensor,
    ) -> Tensor:
        return (
            values * weights
        ).sum() / weights.sum().clamp_min(
            1e-6
        )

    def forward(
        self,
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
        classification_loss: Tensor,
        intervention_scale: float,
        precomputed_feature: Tensor | None = None,
        precomputed_logits: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if self.mode == "tail_soft":
            return self._forward_tail_soft(
                model,
                images,
                targets,
                classification_loss,
                intervention_scale,
                precomputed_feature,
                precomputed_logits,
            )
        feature_map = (
            precomputed_feature
            if precomputed_feature is not None
            else model.forward_features(images)
        )

        original_logits = (
            precomputed_logits
            if precomputed_logits is not None
            else model.forward_head(feature_map)
        )

        attention = getattr(
            model.ml_decoder.decoder.layers[-1],
            "last_attn_weights",
            None,
        )

        if attention is None:
            raise RuntimeError(
                "没有获得 ML-Decoder cross-attention"
            )

        attention = attention.detach().float()

        batch_size = images.shape[0]

        if tuple(attention.shape[:2]) != (
            batch_size,
            self.num_classes,
        ):
            raise RuntimeError(
                "LCI Attention 形状错误："
                f"{tuple(attention.shape)}"
            )

        selected_classes = (
            self._select_positive_class(
                targets
            )
        )

        batch_indices = torch.arange(
            batch_size,
            device=images.device,
        )

        selected_attention = attention[
            batch_indices,
            selected_classes,
        ]

        _, _, height, width = (
            feature_map.shape
        )

        core_mask = self._build_core_mask(
            selected_attention=selected_attention,
            height=height,
            width=width,
        )

        donor_indices, valid_donor_mask = self._select_donors(
            targets=targets,
            selected_classes=selected_classes,
        )

        donor_features = feature_map[
            donor_indices
        ].detach()

        swapped_features = (
            feature_map * core_mask
            + donor_features
            * (1.0 - core_mask)
        )

        context_mask = 1.0 - core_mask

        context_mean = (
            feature_map * context_mask
        ).sum(
            dim=(2, 3),
            keepdim=True,
        ) / context_mask.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1.0)

        core_removed_features = (
            feature_map * context_mask
            + context_mean * core_mask
        )

        swapped_logits = model.forward_head(
            swapped_features
        )

        core_removed_logits = model.forward_head(
            core_removed_features
        )

        selected_original = original_logits[
            batch_indices,
            selected_classes,
        ]

        selected_swapped = swapped_logits[
            batch_indices,
            selected_classes,
        ]

        selected_removed = core_removed_logits[
            batch_indices,
            selected_classes,
        ]

        sample_weights = self.group_weights[
            selected_classes
        ] * valid_donor_mask.to(
            dtype=feature_map.dtype
        )

        swap_positive_per_sample = F.softplus(
            -selected_swapped
        )

        swap_invariance_per_sample = (
            F.smooth_l1_loss(
                selected_swapped,
                selected_original.detach(),
                reduction="none",
            )
        )

        core_drop = (
            selected_original
            - selected_removed
        )

        necessity_per_sample = F.relu(
            self.necessity_margin
            - core_drop
        )

        swap_positive_loss = (
            self._weighted_mean(
                swap_positive_per_sample,
                sample_weights,
            )
        )

        swap_invariance_loss = (
            self._weighted_mean(
                swap_invariance_per_sample,
                sample_weights,
            )
        )

        core_necessity_loss = (
            self._weighted_mean(
                necessity_per_sample,
                sample_weights,
            )
        )

        raw_auxiliary_loss = (
            self.swap_positive_weight
            * swap_positive_loss
            + self.swap_invariance_weight
            * swap_invariance_loss
            + self.core_necessity_weight
            * core_necessity_loss
        )

        scaled_auxiliary_loss = (
            raw_auxiliary_loss
            * float(intervention_scale)
        )

        maximum_auxiliary = (
            classification_loss.detach()
            * self.auxiliary_to_classification_cap
        )

        cap_coefficient = torch.clamp(
            maximum_auxiliary
            / (
                scaled_auxiliary_loss.detach()
                + 1e-8
            ),
            max=1.0,
        )

        effective_auxiliary_loss = (
            scaled_auxiliary_loss
            * cap_coefficient
        )

        selected_groups = self.group_ids[
            selected_classes[valid_donor_mask]
        ]

        selected_group_counts = torch.stack(
            [
                (
                    selected_groups == group_id
                ).sum()
                for group_id in (0, 1, 2)
            ]
        )

        return {
            "logits": original_logits,
            "auxiliary_loss": (
                effective_auxiliary_loss
            ),
            "raw_auxiliary_loss": (
                raw_auxiliary_loss
            ),
            "swap_positive_loss": (
                swap_positive_loss
            ),
            "swap_invariance_loss": (
                swap_invariance_loss
            ),
            "core_necessity_loss": (
                core_necessity_loss
            ),
            "mean_core_drop": (
                (
                    core_drop
                    * valid_donor_mask.to(core_drop.dtype)
                ).sum()
                / valid_donor_mask.sum().clamp_min(1)
            ),
            "mean_core_coverage": (
                (
                    core_mask.mean(dim=(1, 2, 3))
                    * valid_donor_mask.to(core_mask.dtype)
                ).sum()
                / valid_donor_mask.sum().clamp_min(1)
            ),
            "cap_coefficient": (
                cap_coefficient
            ),
            "selected_group_counts": (
                selected_group_counts
            ),
            "feature_map": feature_map,
            "attention": attention,
            "valid_donor_count": valid_donor_mask.sum(),
            "skipped_donor_count": (~valid_donor_mask).sum(),

        }

    def _forward_tail_soft(
        self,
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
        classification_loss: Tensor,
        intervention_scale: float,
        precomputed_feature: Tensor | None = None,
        precomputed_logits: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Intervene on every positive Tail label with a soft object mask."""
        feature_map = (
            precomputed_feature
            if precomputed_feature is not None
            else model.forward_features(images)
        )
        original_logits = (
            precomputed_logits
            if precomputed_logits is not None
            else model.forward_head(feature_map)
        )
        attention = getattr(
            model.ml_decoder.decoder.layers[-1], "last_attn_weights", None
        )
        if attention is None:
            raise RuntimeError("ML-Decoder did not expose class attention")
        attention = attention.detach().float()
        zero = original_logits.sum() * 0.0
        if float(intervention_scale) <= 0.0:
            return {
                "logits": original_logits,
                "auxiliary_loss": zero,
                "raw_auxiliary_loss": zero,
                "swap_positive_loss": zero,
                "swap_invariance_loss": zero,
                "core_necessity_loss": zero,
                "ranking_loss": zero,
                "mean_core_drop": zero,
                "mean_core_coverage": zero,
                "cap_coefficient": zero,
                "selected_group_counts": torch.zeros(
                    3, dtype=torch.long, device=images.device
                ),
                "feature_map": feature_map,
                "attention": attention,
                "valid_donor_count": torch.zeros(
                    (), dtype=torch.long, device=images.device
                ),
                "skipped_donor_count": torch.zeros(
                    (), dtype=torch.long, device=images.device
                ),
            }
        tail_indices = torch.nonzero(self.group_ids == 2, as_tuple=False).flatten()
        positive_pairs = torch.nonzero(
            targets[:, tail_indices] > 0.5, as_tuple=False
        )
        valid_anchor = []
        valid_class = []
        valid_donor = []
        for pair in positive_pairs.tolist():
            anchor_index = int(pair[0])
            class_index = int(tail_indices[int(pair[1])].item())
            candidates = torch.nonzero(
                targets[:, class_index] <= 0.5, as_tuple=False
            ).flatten()
            candidates = candidates[candidates != anchor_index]
            if candidates.numel() == 0:
                continue
            overlap = (
                targets[candidates] * targets[anchor_index].unsqueeze(0)
            ).sum(dim=1)
            donor_index = int(candidates[overlap.argmin()].item())
            valid_anchor.append(anchor_index)
            valid_class.append(class_index)
            valid_donor.append(donor_index)

        if not valid_anchor:
            return {
                "logits": original_logits,
                "auxiliary_loss": zero,
                "raw_auxiliary_loss": zero,
                "swap_positive_loss": zero,
                "swap_invariance_loss": zero,
                "core_necessity_loss": zero,
                "ranking_loss": zero,
                "mean_core_drop": zero,
                "mean_core_coverage": zero,
                "cap_coefficient": zero,
                "selected_group_counts": torch.zeros(
                    3, dtype=torch.long, device=images.device
                ),
                "feature_map": feature_map,
                "attention": attention,
                "valid_donor_count": torch.zeros(
                    (), dtype=torch.long, device=images.device
                ),
                "skipped_donor_count": torch.tensor(
                    positive_pairs.shape[0],
                    dtype=torch.long,
                    device=images.device,
                ),
            }

        anchor_indices = torch.tensor(valid_anchor, device=images.device)
        class_indices = torch.tensor(valid_class, device=images.device)
        donor_indices = torch.tensor(valid_donor, device=images.device)
        selected_attention = attention[anchor_indices, class_indices]
        threshold = torch.quantile(
            selected_attention,
            q=1.0 - self.core_topk_ratio,
            dim=1,
            keepdim=True,
        )
        scale = selected_attention.std(dim=1, keepdim=True).clamp_min(1e-6)
        soft_mask = torch.sigmoid(
            (selected_attention - threshold)
            / (self.soft_mask_temperature * scale)
        )
        height, width = feature_map.shape[-2:]
        soft_mask = soft_mask.reshape(-1, 1, height, width).detach()
        anchor_feature = feature_map[anchor_indices]
        donor_feature = feature_map[donor_indices].detach()
        kept_feature = (
            soft_mask * anchor_feature + (1.0 - soft_mask) * donor_feature
        )
        context_feature = (
            (1.0 - soft_mask) * anchor_feature + soft_mask * donor_feature
        )
        kept_logits = model.forward_head(kept_feature)
        context_logits = model.forward_head(context_feature)
        pair_indices = torch.arange(class_indices.numel(), device=images.device)
        original_score = original_logits[anchor_indices, class_indices]
        kept_score = kept_logits[pair_indices, class_indices]
        context_score = context_logits[pair_indices, class_indices]

        pair_weights = self.inverse_frequency[class_indices]
        pair_weights = pair_weights / pair_weights.mean().clamp_min(1e-6)
        pair_weights = pair_weights.clamp_max(self.maximum_frequency_weight)
        positive_per_pair = F.softplus(-kept_score)
        invariance_per_pair = F.smooth_l1_loss(
            kept_score, original_score.detach(), reduction="none"
        )
        necessity_per_pair = F.softplus(
            self.necessity_margin - original_score.detach() + context_score
        )
        ranking_per_pair = F.softplus(
            self.ranking_margin - kept_score + context_score
        )
        positive_loss = self._weighted_mean(positive_per_pair, pair_weights)
        invariance_loss = self._weighted_mean(invariance_per_pair, pair_weights)
        necessity_loss = self._weighted_mean(necessity_per_pair, pair_weights)
        ranking_loss = self._weighted_mean(ranking_per_pair, pair_weights)
        raw_auxiliary = (
            self.swap_positive_weight * positive_loss
            + self.swap_invariance_weight * invariance_loss
            + self.core_necessity_weight * necessity_loss
            + self.ranking_weight * ranking_loss
        )
        scaled_auxiliary = raw_auxiliary * float(intervention_scale)
        maximum_auxiliary = (
            classification_loss.detach() * self.auxiliary_to_classification_cap
        )
        cap_coefficient = torch.clamp(
            maximum_auxiliary / (scaled_auxiliary.detach() + 1e-8), max=1.0
        )
        effective_auxiliary = scaled_auxiliary * cap_coefficient
        return {
            "logits": original_logits,
            "auxiliary_loss": effective_auxiliary,
            "raw_auxiliary_loss": raw_auxiliary,
            "swap_positive_loss": positive_loss,
            "swap_invariance_loss": invariance_loss,
            "core_necessity_loss": necessity_loss,
            "ranking_loss": ranking_loss,
            "mean_core_drop": (original_score - context_score).mean(),
            "mean_core_coverage": soft_mask.mean(),
            "cap_coefficient": cap_coefficient,
            "selected_group_counts": torch.tensor(
                [0, 0, class_indices.numel()], device=images.device
            ),
            "feature_map": feature_map,
            "attention": attention,
            "valid_donor_count": torch.tensor(
                len(valid_anchor), dtype=torch.long, device=images.device
            ),
            "skipped_donor_count": torch.tensor(
                positive_pairs.shape[0] - len(valid_anchor),
                dtype=torch.long,
                device=images.device,
            ),
        }
