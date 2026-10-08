"""Co-occurrence Evidence Decoupling and Reliability Refinement (CEDR).

CEDR is the public M2 interface. Its two checkpoint-compatible implementation
phases are private to this package and are always executed in order.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._internal.ced import CED as _EvidenceDecouplingImpl
from ._internal.tmr import TMR as _MemoryRefinementImpl


class EvidenceDecoupling(_EvidenceDecouplingImpl):
    """Internal CEDR phase A: split Tail-private and Head-shared evidence."""


class MemoryReliabilityRefinement(_MemoryRefinementImpl):
    """Internal CEDR phase B: verify private evidence with positive/negative memory."""


class CEDR(nn.Module):
    """Unified public M2 wrapper around the two locked internal phases.

    ``residual_alpha`` is the already-selected safe fusion coefficient.  It is
    stored as a buffer so save/load round trips preserve the exact formal
    prediction rule.
    """

    def __init__(
        self,
        evidence_decoupling: EvidenceDecoupling,
        memory_refinement: MemoryReliabilityRefinement,
        residual_alpha: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(residual_alpha) <= 1.0:
            raise ValueError("residual_alpha must be in [0, 1]")
        if not torch.equal(
            evidence_decoupling.tail_indices.cpu(),
            memory_refinement.tail_indices.cpu(),
        ):
            raise ValueError("CEDR internal phases use different Tail indices")
        self.evidence_decoupling = evidence_decoupling
        self.memory_refinement = memory_refinement
        self.register_buffer(
            "residual_alpha", torch.tensor(float(residual_alpha), dtype=torch.float64)
        )

    @property
    def tail_indices(self) -> Tensor:
        return self.evidence_decoupling.tail_indices

    def forward(
        self,
        feature_map: Tensor,
        attention: Tensor,
        baseline_logits: Tensor,
        targets: Tensor | None = None,
        verification_score: Tensor | None = None,
        positive_weights: Tensor | None = None,
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        decoupled = self.evidence_decoupling(
            feature_map, attention, baseline_logits, targets=targets
        )
        separation = 1.0 - F.cosine_similarity(
            decoupled["private_attention"].float(),
            decoupled["shared_attention"].float(),
            dim=2,
            eps=1e-6,
        )
        if verification_score is None:
            verification_score = separation.new_zeros(separation.shape)
        refined = self.memory_refinement(
            decoupled["logits"].float(),
            decoupled["private_feature"].float(),
            shared_features=decoupled["shared_feature"].float(),
            attention_separation=separation,
            verification_score=verification_score,
            targets=targets,
            positive_weights=positive_weights,
        )
        alpha = float(self.residual_alpha.item())
        final_logits = decoupled["logits"].float() + alpha * (
            refined["logits"].float() - decoupled["logits"].float()
        )
        final_tail_residual = alpha * refined["residual"].float()
        result: Dict[str, Tensor | Dict[str, Tensor]] = {
            "logits": final_logits,
            "tail_logits": final_logits[:, self.tail_indices.to(final_logits.device)],
            "private_feature": decoupled["private_feature"],
            "shared_feature": decoupled["shared_feature"],
            "private_attention": decoupled["private_attention"],
            "shared_attention": decoupled["shared_attention"],
            "attention_separation": separation,
            "decoupling_residual": decoupled["residual"],
            "memory_residual": refined["residual"],
            "final_tail_residual": final_tail_residual,
            "decoupling": decoupled,
            "memory_refinement": refined,
        }
        return result


__all__ = ["CEDR"]
