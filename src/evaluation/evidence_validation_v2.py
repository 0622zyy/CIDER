from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def _np(value: Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def safe_mean(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def binary_curve(targets: Sequence[float], scores: Sequence[float]) -> dict[str, np.ndarray | float]:
    y = _np(targets).astype(bool)
    s = _np(scores).astype(float)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        empty = np.asarray([], dtype=float)
        return {"threshold": empty, "recall": empty, "fpr": empty, "precision": empty,
                "auroc": float("nan"), "auprc": float("nan")}
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct].astype(float)
    fp = np.cumsum(~y)[distinct].astype(float)
    recall = tp / positives
    fpr = fp / negatives
    precision = tp / np.maximum(tp + fp, 1.0)
    threshold = s[distinct]
    roc_x = np.r_[0.0, fpr, 1.0]
    roc_y = np.r_[0.0, recall, 1.0]
    auroc = float(np.trapz(roc_y, roc_x))
    # Average precision: right-continuous precision weighted by recall increments.
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {"threshold": threshold, "recall": recall, "fpr": fpr,
            "precision": precision, "auroc": auroc, "auprc": auprc}


def _at_target(x: np.ndarray, y: np.ndarray, target: float, mode: str) -> float:
    if not x.size:
        return float("nan")
    if mode == "at_least":
        indices = np.flatnonzero(x >= target)
        return float(y[indices[0]]) if indices.size else float("nan")
    indices = np.flatnonzero(x <= target)
    return float(y[indices[-1]]) if indices.size else float("nan")


def operating_points(targets: Sequence[float], scores: Sequence[float],
                     recall_targets: Sequence[float] = (0.6,),
                     fpr_targets: Sequence[float] = (0.005,)) -> list[dict[str, float | str]]:
    curve = binary_curve(targets, scores)
    recall = np.asarray(curve["recall"])
    fpr = np.asarray(curve["fpr"])
    precision = np.asarray(curve["precision"])
    rows: list[dict[str, float | str]] = []
    for target in recall_targets:
        rows.append({
            "constraint": "fixed_recall", "target": float(target),
            "fpr": _at_target(recall, fpr, float(target), "at_least"),
            "recall": float(target),
            "precision": _at_target(recall, precision, float(target), "at_least"),
        })
    for target in fpr_targets:
        rows.append({
            "constraint": "fixed_fpr", "target": float(target),
            "fpr": float(target),
            "recall": _at_target(fpr, recall, float(target), "at_most"),
            "precision": _at_target(fpr, precision, float(target), "at_most"),
        })
    return rows


def calibration_metrics(targets: Sequence[float], probabilities: Sequence[float], bins: int = 15) -> dict[str, float]:
    y = _np(targets).astype(float)
    p = np.clip(_np(probabilities).astype(float), 0.0, 1.0)
    finite = np.isfinite(p) & np.isfinite(y)
    y, p = y[finite], p[finite]
    if not y.size:
        return {"ece": float("nan"), "brier_score": float("nan"), "best_f1_threshold": float("nan"), "best_f1": float("nan")}
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    membership = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = membership == index
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        pred = p >= threshold
        tp = float(np.sum(pred & (y > 0.5)))
        fp = float(np.sum(pred & (y <= 0.5)))
        fn = float(np.sum(~pred & (y > 0.5)))
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return {"ece": float(ece), "brier_score": float(np.mean((p - y) ** 2)),
            "best_f1_threshold": best_threshold, "best_f1": float(best_f1)}


def bootstrap_ci(values: Sequence[float], seed: int = 42, samples: int = 2000,
                 confidence: float = 0.95) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=float)
    for start in range(0, int(samples), 256):
        count = min(256, int(samples) - start)
        indices = rng.integers(0, x.size, size=(count, x.size))
        means[start:start + count] = x[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.quantile(means, [alpha, 1.0 - alpha]))


def paired_permutation_test(first: Sequence[float], second: Sequence[float], seed: int = 42,
                            permutations: int = 10000) -> float:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    difference = a[keep] - b[keep]
    if not difference.size:
        return float("nan")
    observed = abs(float(difference.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, int(permutations), 512):
        count = min(512, int(permutations) - start)
        signs = rng.choice((-1.0, 1.0), size=(count, difference.size))
        extreme += int(np.sum(np.abs((signs * difference).mean(axis=1)) >= observed))
    return float((extreme + 1) / (int(permutations) + 1))


def attention_overlap(first: Tensor, second: Tensor) -> Tensor:
    """Asymmetric overlap requested by the v2 protocol: sum(min)/sum(Tail)."""
    first, second = first.float(), second.float()
    return torch.minimum(first, second).sum(dim=-1) / first.sum(dim=-1).clamp_min(1e-8)


def topq_iou(first: Tensor, second: Tensor, ratio: float = 0.15) -> Tensor:
    first, second = first.float(), second.float()
    count = first.shape[-1]
    topk = max(1, min(count, int(round(count * float(ratio)))))
    first_mask = torch.zeros_like(first, dtype=torch.bool)
    second_mask = torch.zeros_like(second, dtype=torch.bool)
    first_mask.scatter_(-1, first.topk(topk, dim=-1).indices, True)
    second_mask.scatter_(-1, second.topk(topk, dim=-1).indices, True)
    intersection = (first_mask & second_mask).sum(dim=-1).float()
    union = (first_mask | second_mask).sum(dim=-1).float()
    return intersection / union.clamp_min(1.0)


def cohens_d(positive: Sequence[float], negative: Sequence[float]) -> float:
    a, b = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return float("nan"), float("nan")
    ra, rb = rankdata(a), rankdata(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    # Large-sample t approximation; documented as approximate for small Tail sets.
    if abs(rho) >= 1:
        return rho, 0.0
    t = abs(rho) * np.sqrt((a.size - 2) / max(1.0 - rho * rho, 1e-12))
    # Normal approximation avoids a scipy dependency in the project environment.
    p = float(math.erfc(t / np.sqrt(2.0)))
    return rho, p


def active_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in pairs if bool(row.get("active_pair", False))]
