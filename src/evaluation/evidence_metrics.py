from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from src.utils.metrics import binary_average_precision


def _targets(dataset: Any) -> Tensor:
    """Read labels already parsed by the official project dataset class."""
    return torch.stack([sample.target.float() for sample in dataset.samples])


def build_train_pairs(
    train_dataset: Any,
    tail_indices: Sequence[int],
    head_indices: Sequence[int],
    label_names: Sequence[str],
    topk: int = 3,
) -> list[dict[str, Any]]:
    """Select Tail-Head references using official training labels only.

    Ranking exactly follows the CED training implementation: P(Head|Tail),
    then Head frequency, then class index. Zero-count fallback entries are kept
    for an auditable Top-K table, but marked inactive for aggregate reporting.
    """
    targets = _targets(train_dataset)
    frequency = targets.sum(dim=0)
    rows: list[dict[str, Any]] = []
    topk = max(int(topk), 1)
    for tail_index in sorted(int(v) for v in tail_indices):
        tail_positive = targets[:, tail_index] > 0.5
        tail_count = int(tail_positive.sum().item())
        scored = []
        for head_index in sorted(int(v) for v in head_indices):
            count = int(
                (tail_positive & (targets[:, head_index] > 0.5)).sum().item()
            )
            conditional = count / max(tail_count, 1)
            scored.append(
                (conditional, int(frequency[head_index].item()), head_index, count)
            )
        scored.sort(reverse=True)
        for rank, (conditional, head_frequency, head_index, count) in enumerate(
            scored[:topk], start=1
        ):
            rows.append(
                {
                    "tail_index": tail_index,
                    "tail_name": label_names[tail_index],
                    "head_index": head_index,
                    "head_name": label_names[head_index],
                    "rank": rank,
                    "train_cooccurrence_count": count,
                    "train_tail_count": tail_count,
                    "train_head_count": head_frequency,
                    "train_p_head_given_tail": conditional,
                    "active_pair": bool(count > 0),
                    "selection_source": "official_train_labels_only",
                }
            )
    return rows


def bhattacharyya_overlap(first: Tensor, second: Tensor) -> Tensor:
    """Bhattacharyya overlap used by CED after spatial normalization."""
    first = first.float() / first.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    second = second.float() / second.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.sqrt(first.clamp_min(1e-8) * second.clamp_min(1e-8)).sum(dim=-1)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_ap(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if targets.size == 0 or np.unique(targets > 0.5).size < 2:
        return float("nan")
    return 100.0 * binary_average_precision(targets, probabilities)


def conditional_subset_rows(
    method: str,
    targets: Tensor,
    probabilities: Tensor,
    pairs: Sequence[Mapping[str, Any]],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Compute pair-wise conditioned diagnostics without selecting on eval labels.

    AP on a positive-only subset is undefined. Therefore de-cooccur AP is
    evaluated in the Head-absent cohort and co-present AP in the Head-present
    cohort; the positive subset names are retained for recall/probability.
    """
    y = targets.detach().cpu().numpy()
    p = probabilities.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        tail = int(pair["tail_index"])
        head = int(pair["head_index"])
        tail_positive = y[:, tail] > 0.5
        head_positive = y[:, head] > 0.5
        co_positive = tail_positive & head_positive
        de_positive = tail_positive & ~head_positive
        hard_negative = ~tail_positive & head_positive
        head_present = head_positive
        head_absent = ~head_positive
        pred = p[:, tail] >= float(threshold)
        co_recall = _safe_mean(pred[co_positive].astype(float))
        de_recall = _safe_mean(pred[de_positive].astype(float))
        hard_fpr = _safe_mean(pred[hard_negative].astype(float))
        predicted_in_head_cohort = int(pred[head_present].sum())
        hard_precision = (
            100.0 * int((pred & co_positive).sum()) / predicted_in_head_cohort
            if predicted_in_head_cohort
            else float("nan")
        )
        co_ap = _safe_ap(y[head_present, tail], p[head_present, tail])
        de_ap = _safe_ap(y[head_absent, tail], p[head_absent, tail])
        row = dict(pair)
        row.update(
            {
                "method": method,
                "threshold": float(threshold),
                "co_present_positive_count": int(co_positive.sum()),
                "de_cooccur_positive_count": int(de_positive.sum()),
                "cooccur_hard_negative_count": int(hard_negative.sum()),
                "co_present_tail_recall": 100.0 * co_recall,
                "co_present_tail_ap": co_ap,
                "co_present_mean_probability": 100.0 * _safe_mean(p[co_positive, tail]),
                "de_cooccur_tail_recall": 100.0 * de_recall,
                "de_cooccur_tail_ap": de_ap,
                "de_cooccur_mean_probability": 100.0 * _safe_mean(p[de_positive, tail]),
                "hard_negative_tail_fpr": 100.0 * hard_fpr,
                "hard_negative_tail_precision": hard_precision,
                "hard_negative_mean_probability": 100.0 * _safe_mean(p[hard_negative, tail]),
                "recall_gap_co_present_minus_de": 100.0 * (co_recall - de_recall),
                "ap_gap_co_present_minus_de": co_ap - de_ap,
                "probability_gap_co_present_minus_de": 100.0
                * (_safe_mean(p[co_positive, tail]) - _safe_mean(p[de_positive, tail])),
                "co_present_ap_scope": "evaluation samples with Head=1",
                "de_cooccur_ap_scope": "evaluation samples with Head=0",
            }
        )
        rows.append(row)
    return rows


def aggregate_subset_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Macro-average active, positive-count train pairs for each method."""
    fields = (
        "co_present_tail_recall",
        "co_present_tail_ap",
        "co_present_mean_probability",
        "de_cooccur_tail_recall",
        "de_cooccur_tail_ap",
        "de_cooccur_mean_probability",
        "hard_negative_tail_fpr",
        "hard_negative_tail_precision",
        "hard_negative_mean_probability",
        "recall_gap_co_present_minus_de",
        "ap_gap_co_present_minus_de",
        "probability_gap_co_present_minus_de",
    )
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    result = []
    for method in methods:
        selected = [
            row for row in rows if row["method"] == method and bool(row["active_pair"])
        ]
        aggregate: dict[str, Any] = {
            "method": method,
            "row_type": "active_pair_macro",
            "active_pair_count": len(selected),
        }
        for field in fields:
            values = np.asarray([float(row[field]) for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            aggregate[field] = float(np.mean(finite)) if finite.size else float("nan")
        result.append(aggregate)
    return result
