from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(
                file,
                encoding="latin1",
            )


def _to_index_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
    elif hasattr(value, "tolist"):
        sequence = value.tolist()
    else:
        raise TypeError(
            "类别划分无法转换为索引列表："
            f"type={type(value).__name__}"
        )

    if not isinstance(sequence, list):
        sequence = [sequence]

    return sorted(
        {
            int(index)
            for index in sequence
        }
    )


def load_class_split(
    path: str | Path,
    num_classes: int,
) -> Dict[str, list[int]]:
    split_path = Path(path)

    if not split_path.is_file():
        raise FileNotFoundError(
            f"class_split 文件不存在：{split_path}"
        )

    raw_split = _load_pickle(split_path)

    if not isinstance(raw_split, Mapping):
        raise TypeError(
            "class_split.pkl 应为字典，"
            f"实际类型={type(raw_split).__name__}"
        )

    aliases = {
        "head": {
            "head",
            "many",
            "many_shot",
            "manyshot",
        },
        "middle": {
            "middle",
            "medium",
            "med",
            "medium_shot",
            "mediumshot",
        },
        "tail": {
            "tail",
            "few",
            "few_shot",
            "fewshot",
        },
    }

    normalized_keys = {
        str(key)
        .lower()
        .replace("-", "_")
        .replace(" ", "_"): key
        for key in raw_split.keys()
    }

    result: Dict[str, list[int]] = {}

    for canonical_name, possible_names in aliases.items():
        matched_key = None

        for normalized_name, original_key in (
            normalized_keys.items()
        ):
            if normalized_name in possible_names:
                matched_key = original_key
                break

        if matched_key is None:
            raise KeyError(
                f"未在 class_split.pkl 中找到 "
                f"{canonical_name}，"
                f"现有键={list(raw_split.keys())}"
            )

        indices = _to_index_list(
            raw_split[matched_key]
        )

        invalid_indices = [
            index
            for index in indices
            if index < 0 or index >= num_classes
        ]

        if invalid_indices:
            raise IndexError(
                f"{canonical_name} 中存在越界索引："
                f"{invalid_indices}"
            )

        result[canonical_name] = indices

    head_set = set(result["head"])
    middle_set = set(result["middle"])
    tail_set = set(result["tail"])

    if head_set & middle_set:
        raise ValueError(
            "head 与 middle 存在重叠："
            f"{sorted(head_set & middle_set)}"
        )

    if head_set & tail_set:
        raise ValueError(
            "head 与 tail 存在重叠："
            f"{sorted(head_set & tail_set)}"
        )

    if middle_set & tail_set:
        raise ValueError(
            "middle 与 tail 存在重叠："
            f"{sorted(middle_set & tail_set)}"
        )

    covered = (
        head_set
        | middle_set
        | tail_set
    )

    expected = set(range(num_classes))

    if covered != expected:
        raise ValueError(
            "Head/Middle/Tail 未完整覆盖类别。\n"
            f"缺失={sorted(expected - covered)}\n"
            f"额外={sorted(covered - expected)}"
        )

    return result


def binary_average_precision(
    targets: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    计算单个类别的非插值 Average Precision。

    返回范围：
        [0, 1]

    当该类别没有正样本时返回 NaN。
    """

    targets = np.asarray(
        targets,
        dtype=np.float64,
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if targets.shape != scores.shape:
        raise ValueError(
            "targets 与 scores 长度不一致："
            f"{targets.shape} != {scores.shape}"
        )

    binary_targets = (
        targets > 0.5
    ).astype(np.float64)

    positive_count = int(
        binary_targets.sum()
    )

    if positive_count == 0:
        return float("nan")

    # mergesort 保证分数相同时排序稳定。
    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_targets = binary_targets[order]

    true_positives = np.cumsum(
        sorted_targets
    )

    ranks = np.arange(
        1,
        len(sorted_targets) + 1,
        dtype=np.float64,
    )

    precision = true_positives / ranks

    average_precision = (
        precision * sorted_targets
    ).sum() / positive_count

    return float(average_precision)


def per_class_average_precision(
    targets: np.ndarray | Tensor,
    scores: np.ndarray | Tensor,
) -> np.ndarray:
    if isinstance(targets, Tensor):
        targets = (
            targets.detach()
            .cpu()
            .numpy()
        )

    if isinstance(scores, Tensor):
        scores = (
            scores.detach()
            .cpu()
            .numpy()
        )

    targets = np.asarray(targets)
    scores = np.asarray(scores)

    if targets.shape != scores.shape:
        raise ValueError(
            "targets 与 scores 形状不一致："
            f"{targets.shape} != {scores.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            "mAP 输入应为二维矩阵 [N,C]，"
            f"实际 shape={targets.shape}"
        )

    num_classes = targets.shape[1]

    aps = np.empty(
        num_classes,
        dtype=np.float64,
    )

    for class_index in range(num_classes):
        aps[class_index] = binary_average_precision(
            targets[:, class_index],
            scores[:, class_index],
        )

    return aps


def _safe_nanmean(
    values: Sequence[float],
) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = array[~np.isnan(array)]

    if valid.size == 0:
        return float("nan")

    return float(valid.mean())


def evaluate_multilabel_map(
    targets: np.ndarray | Tensor,
    scores: np.ndarray | Tensor,
    class_split: Mapping[str, Iterable[int]],
) -> Dict[str, Any]:
    aps = per_class_average_precision(
        targets=targets,
        scores=scores,
    )

    num_classes = len(aps)

    expected_groups = (
        "head",
        "middle",
        "tail",
    )

    metrics: Dict[str, Any] = {
        "overall_map": (
            100.0 * _safe_nanmean(aps)
        ),
        "per_class_ap": (
            100.0 * aps
        ),
        "num_classes": num_classes,
        "num_valid_classes": int(
            np.sum(~np.isnan(aps))
        ),
    }

    for group_name in expected_groups:
        if group_name not in class_split:
            raise KeyError(
                f"class_split 缺少 {group_name}"
            )

        indices = [
            int(index)
            for index in class_split[group_name]
        ]

        metrics[f"{group_name}_map"] = (
            100.0
            * _safe_nanmean(
                aps[indices]
            )
        )

        metrics[
            f"{group_name}_indices"
        ] = indices

    return metrics
