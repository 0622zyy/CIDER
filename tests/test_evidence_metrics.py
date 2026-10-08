from types import SimpleNamespace

import torch

from src.evaluation.evidence_metrics import (
    aggregate_subset_rows,
    build_train_pairs,
    conditional_subset_rows,
)


def _dataset(rows):
    return SimpleNamespace(
        samples=[SimpleNamespace(target=torch.tensor(row).float()) for row in rows]
    )


def test_train_pairs_use_training_counts_and_mark_zero_fallback():
    dataset = _dataset(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    pairs = build_train_pairs(dataset, [0, 1], [2, 3], ["t0", "t1", "h2", "h3"], 2)
    assert pairs[0]["head_index"] == 2
    assert pairs[0]["train_cooccurrence_count"] == 2
    assert pairs[1]["active_pair"] is False


def test_subset_ap_is_conditioned_not_positive_only():
    targets = torch.tensor(
        [
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0],
        ]
    ).float()
    probabilities = torch.tensor(
        [
            [0.8, 0.9],
            [0.6, 0.1],
            [0.9, 0.8],
            [0.2, 0.2],
        ]
    )
    pair = {
        "tail_index": 0,
        "tail_name": "tail",
        "head_index": 1,
        "head_name": "head",
        "rank": 1,
        "active_pair": True,
    }
    row = conditional_subset_rows("model", targets, probabilities, [pair])[0]
    assert row["co_present_positive_count"] == 1
    assert row["de_cooccur_positive_count"] == 1
    assert row["cooccur_hard_negative_count"] == 1
    assert row["co_present_tail_ap"] == 50.0
    assert row["de_cooccur_tail_ap"] == 100.0
    assert aggregate_subset_rows([row])[0]["active_pair_count"] == 1
