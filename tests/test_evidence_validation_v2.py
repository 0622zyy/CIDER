import numpy as np
import torch

from src.evaluation.evidence_validation_v2 import (
    attention_overlap,
    binary_curve,
    bootstrap_ci,
    calibration_metrics,
    cohens_d,
    operating_points,
    paired_permutation_test,
    spearman,
    topq_iou,
)


def test_curve_and_operating_points_are_scale_free():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    curve = binary_curve(y, p)
    assert np.isclose(curve["auroc"], 0.75)
    assert np.isclose(curve["auprc"], 5 / 6)
    fixed_recall = operating_points(y, p, (0.5,), (0.5,))[0]
    assert fixed_recall["fpr"] == 0.0
    assert fixed_recall["precision"] == 1.0


def test_calibration_and_effect_statistics():
    result = calibration_metrics([0, 1], [0.0, 1.0], bins=2)
    assert result["ece"] == 0.0
    assert result["brier_score"] == 0.0
    assert cohens_d([2, 3, 4], [-2, -3, -4]) > 4
    low, high = bootstrap_ci([1, 2, 3], samples=100, seed=1)
    assert low <= 2 <= high
    assert paired_permutation_test([2, 3, 4], [1, 2, 3], permutations=100, seed=1) <= 0.3


def test_attention_metrics_and_spearman():
    a = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert torch.allclose(attention_overlap(a, b), torch.tensor([0.5]))
    assert torch.allclose(topq_iou(a, b, 0.5), torch.tensor([1 / 3]))
    rho, p = spearman([1, 2, 3, 4], [2, 4, 6, 8])
    assert np.isclose(rho, 1.0)
    assert p == 0.0
