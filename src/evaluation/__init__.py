"""Evaluation-only diagnostics for co-occurrence evidence substitution."""

from .evidence_metrics import (
    aggregate_subset_rows,
    bhattacharyya_overlap,
    build_train_pairs,
    conditional_subset_rows,
)

__all__ = [
    "aggregate_subset_rows",
    "bhattacharyya_overlap",
    "build_train_pairs",
    "conditional_subset_rows",
]
