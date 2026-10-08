from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


Reduction = Literal["none", "mean", "sum"]


class AsymmetricLoss(nn.Module):
    """
    多标签分类常用的 Asymmetric Loss。

    这里只作为 baseline 损失函数，不属于本项目创新。

    默认配置：
        gamma_negative = 4
        gamma_positive = 0
        clip = 0.05

    reduction="mean" 表示：
        先对每个样本的所有类别求和，
        再对 batch 求平均。
    """

    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: Reduction = "mean",
        detach_focal_weight: bool = True,
    ) -> None:
        super().__init__()

        if gamma_negative < 0:
            raise ValueError("gamma_negative 不能小于 0")

        if gamma_positive < 0:
            raise ValueError("gamma_positive 不能小于 0")

        if clip < 0:
            raise ValueError("clip 不能小于 0")

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"不支持的 reduction：{reduction}"
            )

        self.gamma_negative = float(gamma_negative)
        self.gamma_positive = float(gamma_positive)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction
        self.detach_focal_weight = bool(
            detach_focal_weight
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "logits 与 targets 形状必须一致："
                f"logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        if logits.ndim != 2:
            raise ValueError(
                "ASL 期望二维输入 [B,C]，"
                f"实际 shape={tuple(logits.shape)}"
            )

        targets = targets.to(
            dtype=logits.dtype,
            device=logits.device,
        )

        probabilities_positive = torch.sigmoid(logits)
        probabilities_negative = (
            1.0 - probabilities_positive
        )

        if self.clip > 0:
            probabilities_negative = (
                probabilities_negative + self.clip
            ).clamp(max=1.0)

        positive_log_likelihood = (
            targets
            * torch.log(
                probabilities_positive.clamp(
                    min=self.eps
                )
            )
        )

        negative_log_likelihood = (
            (1.0 - targets)
            * torch.log(
                probabilities_negative.clamp(
                    min=self.eps
                )
            )
        )

        loss = (
            positive_log_likelihood
            + negative_log_likelihood
        )

        if (
            self.gamma_negative > 0
            or self.gamma_positive > 0
        ):
            positive_probability = (
                probabilities_positive * targets
            )

            negative_probability = (
                probabilities_negative
                * (1.0 - targets)
            )

            gamma = (
                self.gamma_positive * targets
                + self.gamma_negative
                * (1.0 - targets)
            )

            if self.detach_focal_weight:
                with torch.no_grad():
                    focal_weight = torch.pow(
                        1.0
                        - positive_probability
                        - negative_probability,
                        gamma,
                    )
            else:
                focal_weight = torch.pow(
                    1.0
                    - positive_probability
                    - negative_probability,
                    gamma,
                )

            loss = loss * focal_weight

        # 对类别维度求和，每个样本得到一个损失。
        sample_loss = -loss.sum(dim=1)

        if self.reduction == "none":
            return sample_loss

        if self.reduction == "sum":
            return sample_loss.sum()

        return sample_loss.mean()
