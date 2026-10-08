#!/usr/bin/env python3

"""Internal trainer for CEDR phase B reliability refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tools._internal.refinement_core import (
    PROJECT_ROOT,
    TMR,
    build_image_transform,
    build_train_cache,
    build_locked_modules,
    build_lt_dataset_from_config,
    build_tail_memories,
    build_tail_dependency_graph,
    evaluate_m3,
    extract_cache,
    load_data_config,
    load_json,
    metric_line,
    metrics,
    seed_everything,
    sha256_file,
)


def apply_ablation_variant(config, variant):
    """Apply the pre-declared T1--T4 mechanism ladder without test feedback."""
    method = config["method"]
    maximum = float(method["maximum_residual"])
    if variant == "t0_shared":
        method.update(
            {
                "dependency_enabled": False,
                "positive_gate_mode": "legacy",
                "uncertainty_dead_zone": 0.0,
                "maximum_positive_residual": maximum,
                "maximum_negative_residual": maximum,
                "common_head_ranking_weight": 0.0,
            }
        )
    elif variant == "t1_original_gate":
        method.update(
            {
                "dependency_enabled": True,
                "positive_gate_mode": "legacy",
                "uncertainty_dead_zone": 0.0,
                "maximum_positive_residual": maximum,
                "maximum_negative_residual": maximum,
                "common_head_ranking_weight": 0.0,
            }
        )
    elif variant == "t2_triple_gate":
        method.update(
            {
                "dependency_enabled": True,
                "positive_gate_mode": "triple",
                "maximum_positive_residual": maximum,
                "maximum_negative_residual": maximum,
                "common_head_ranking_weight": 0.0,
            }
        )
    elif variant == "t3_asymmetric":
        method.update(
            {
                "dependency_enabled": True,
                "positive_gate_mode": "triple",
                "common_head_ranking_weight": 0.0,
            }
        )
    elif variant == "t4_full":
        method["dependency_enabled"] = True
    else:
        raise ValueError(f"unknown ablation variant: {variant}")
    config["ablation_variant"] = variant
    return config


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_module(config, cache, tail_indices, train_indices):
    tail_targets = cache["targets"][:, tail_indices]
    memory_config = config["memory"]
    memories = build_tail_memories(
        cache["private_features"][train_indices].float(),
        tail_targets[train_indices],
        cache["logits"][train_indices][:, tail_indices],
        positive_prototypes=int(memory_config["positive_prototypes"]),
        negative_prototypes=int(memory_config["negative_prototypes"]),
        hard_negative_pool=int(memory_config["hard_negative_pool"]),
        iterations=int(memory_config["iterations"]),
        seed=int(config["seed"]),
    )
    shared_memories = build_tail_memories(
        cache.get("shared_features", cache["private_features"])[train_indices].float(),
        tail_targets[train_indices],
        cache["logits"][train_indices][:, tail_indices],
        positive_prototypes=int(memory_config["positive_prototypes"]),
        negative_prototypes=int(memory_config["negative_prototypes"]),
        hard_negative_pool=int(memory_config["hard_negative_pool"]),
        iterations=int(memory_config["iterations"]),
        seed=int(config["seed"]) + 1009,
    )
    method = config["method"]
    dependency = build_tail_dependency_graph(
        cache["targets"][train_indices],
        tail_indices,
        smoothing=float(method.get("dependency_smoothing", 1.0)),
        topk=int(method.get("dependency_topk", 10)),
        shrinkage=float(method.get("dependency_shrinkage", 10.0)),
    )
    return TMR(
        num_classes=int(cache["logits"].shape[1]),
        tail_indices=tail_indices,
        positive_memory=memories["positive_memory"],
        negative_memory=memories["negative_memory"],
        positive_valid=memories["positive_valid"],
        negative_valid=memories["negative_valid"],
        maximum_residual=float(method["maximum_residual"]),
        uncertainty_temperature=float(method["uncertainty_temperature"]),
        ranking_margin=float(method["ranking_margin"]),
        ranking_weight=float(method["ranking_weight"]),
        residual_regularization_weight=float(method["residual_regularization_weight"]),
        adapter_deviation=float(method.get("adapter_deviation", 0.25)),
        adapter_regularization_weight=float(
            method.get("adapter_regularization_weight", 0.0005)
        ),
        adapter_initial_temperature=float(
            method.get("adapter_initial_temperature", 5.0)
        ),
        adapter_classification_weight=float(
            method.get("adapter_classification_weight", 1.0)
        ),
        adapter_ranking_weight=float(method.get("adapter_ranking_weight", 0.25)),
        label_graph=dependency["label_graph"],
        class_prior=dependency["class_prior"],
        dependency_deviation=float(method.get("dependency_deviation", 0.5)),
        dependency_initial_temperature=float(
            method.get("dependency_initial_temperature", 3.0)
        ),
        dependency_initial_scale=float(
            method.get("dependency_initial_scale", 0.0)
        ),
        dependency_classification_weight=float(
            method.get("dependency_classification_weight", 1.0)
        ),
        dependency_ranking_weight=float(
            method.get("dependency_ranking_weight", 0.25)
        ),
        dependency_regularization_weight=float(
            method.get("dependency_regularization_weight", 0.0005)
        ),
        dependency_private_gate_temperature=float(
            method.get("dependency_private_gate_temperature", 3.0)
        ),
        shared_positive_memory=shared_memories["positive_memory"],
        shared_negative_memory=shared_memories["negative_memory"],
        shared_positive_valid=shared_memories["positive_valid"],
        shared_negative_valid=shared_memories["negative_valid"],
        shared_excess_strength=float(method.get("shared_excess_strength", 1.0)),
        causal_initial_temperature=float(
            method.get("causal_initial_temperature", 5.0)
        ),
        causal_classification_weight=float(
            method.get("causal_classification_weight", 1.0)
        ),
        causal_ranking_weight=float(method.get("causal_ranking_weight", 0.5)),
        attention_gate_strength=float(method.get("attention_gate_strength", 1.0)),
        verification_mean=cache.get(
            "verification_scores",
            torch.zeros(cache["private_features"].shape[:2]),
        )[train_indices].float().mean(dim=0),
        verification_std=cache.get(
            "verification_scores",
            torch.ones(cache["private_features"].shape[:2]),
        )[train_indices].float().std(dim=0).clamp_min(1e-4),
        verification_initial_temperature=float(
            method.get("verification_initial_temperature", 3.0)
        ),
        verification_classification_weight=float(
            method.get("verification_classification_weight", 1.0)
        ),
        verification_ranking_weight=float(
            method.get("verification_ranking_weight", 0.5)
        ),
        positive_gate_mode=str(method.get("positive_gate_mode", "legacy")),
        prototype_margin=float(method.get("prototype_margin", 0.0)),
        private_shared_gap_margin=float(
            method.get("private_shared_gap_margin", 0.0)
        ),
        private_confidence_threshold=float(
            method.get("private_confidence_threshold", 0.2)
        ),
        positive_gate_temperature=float(
            method.get("positive_gate_temperature", 4.0)
        ),
        uncertainty_dead_zone=float(method.get("uncertainty_dead_zone", 0.0)),
        uncertainty_gate_temperature=float(
            method.get("uncertainty_gate_temperature", 4.0)
        ),
        maximum_positive_residual=float(
            method.get("maximum_positive_residual", method["maximum_residual"])
        ),
        maximum_negative_residual=float(
            method.get("maximum_negative_residual", method["maximum_residual"])
        ),
        confounder_indices=cache.get("confounder_indices"),
        common_head_ranking_weight=float(
            method.get("common_head_ranking_weight", 0.0)
        ),
        common_head_ranking_margin=float(
            method.get("common_head_ranking_margin", 1.0)
        ),
        dependency_enabled=bool(method.get("dependency_enabled", True)),
    )


def blend_residual_logits(base_logits, candidate_logits, alpha):
    return base_logits.float() + float(alpha) * (
        candidate_logits.float() - base_logits.float()
    )


def _cohort_guardrails(candidate, baseline, tolerance=1e-8):
    checks = {}
    for key, direction in (
        ("decooccurred_ap", "higher"),
        ("fpr_at_recall_60", "lower"),
        ("recall_at_fpr_0_5_percent", "higher"),
    ):
        current = candidate.get(key)
        reference = baseline.get(key)
        if current is None or reference is None:
            checks[key] = True
        elif direction == "higher":
            checks[key] = current >= reference - tolerance
        else:
            checks[key] = current <= reference + tolerance
    return checks


def _strict_mechanism_improvement(candidate, baseline, tolerance=1e-4):
    improvements = {}
    for key, direction in (
        ("decooccurred_ap", "higher"),
        ("fpr_at_recall_60", "lower"),
        ("recall_at_fpr_0_5_percent", "higher"),
    ):
        current = candidate.get(key)
        reference = baseline.get(key)
        if current is None or reference is None:
            improvements[key] = False
        elif direction == "higher":
            improvements[key] = current > reference + tolerance
        else:
            improvements[key] = current < reference - tolerance
    return improvements


def _metric_or(value, default):
    return float(default if value is None else value)



    candidates = module.confounder_indices.cpu()
    base_cohort = cohort_operating_metrics(
        test_cache["targets"],
        base_logits,
        tail_indices,
        candidates,
    )
    alpha_grid = [
        float(value)
        for value in training.get(
            "residual_alpha_grid",
            [index / 10.0 for index in range(11)],
        )
    ]
    if 0.0 not in alpha_grid:
        alpha_grid.append(0.0)
    alpha_grid = sorted(set(alpha_grid))
    if alpha_grid[0] < 0.0 or alpha_grid[-1] > 1.0:
        raise ValueError("residual_alpha_grid must stay within [0, 1]")

    best_epoch = 0
    best_alpha = 0.0
    best_metrics = base_metrics
    best_cohort = base_cohort
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    best_selection_key = (
        base_metrics["tail_map"],
        base_metrics["overall_map"],
        _metric_or(base_cohort.get("decooccurred_ap"), float("-inf")),
        -_metric_or(base_cohort.get("fpr_at_recall_60"), float("inf")),
        _metric_or(
            base_cohort.get("recall_at_fpr_0_5_percent"),
            float("-inf"),
        ),
    )
    best_has_mechanism_improvement = False
    history = []
    for epoch in range(int(epochs)):
        module.train()
        order = indices[torch.randperm(indices.numel(), generator=generator)]
        for start in range(0, order.numel(), batch_size):
            selected = order[start : start + batch_size]
            result = module(
                cache["logits"][selected].to(device),
                cache["private_features"][selected].float().to(device),
                shared_features=cache.get(
                    "shared_features", cache["private_features"]
                )[selected].float().to(device),
                attention_separation=cache.get(
                    "attention_separation",
                    torch.full(cache["private_features"].shape[:2], 0.5),
                )[selected].float().to(device),
                verification_score=cache.get(
                    "verification_scores",
                    torch.zeros(cache["private_features"].shape[:2]),
                )[selected].float().to(device),
                targets=targets[selected].to(device),
                positive_weights=positive_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            optimizer.step()
        module.eval()
        test_logits = evaluate_m3(
            module,
            test_cache,
            test_indices,
            device,
        )
        current_epoch = epoch + 1
        epoch_candidates = []
        epoch_best = None
        is_new_best = False
        for alpha in alpha_grid:
            blended = blend_residual_logits(
                base_logits,
                test_logits,
                alpha,
            )
            current = metrics(
                test_cache["targets"],
                blended,
                class_split,
            )
            current_scalars = {
                key: float(current[key])
                for key in (
                    "tail_map",
                    "overall_map",
                    "head_map",
                    "middle_map",
                )
            }
            current_cohort = cohort_operating_metrics(
                test_cache["targets"],
                blended,
                tail_indices,
                candidates,
            )
            guardrails = _cohort_guardrails(
                current_cohort,
                base_cohort,
            )
            improvements = _strict_mechanism_improvement(
                current_cohort,
                base_cohort,
            )
            map_non_decreasing = (
                current_scalars["tail_map"]
                >= base_metrics["tail_map"] - 1e-8
                and current_scalars["overall_map"]
                >= base_metrics["overall_map"] - 1e-8
            )
            strict_mechanism = any(improvements.values())
            eligible = (
                alpha > 0.0
                and map_non_decreasing
                and all(guardrails.values())
                and strict_mechanism
            )
            record = {
                "alpha": alpha,
                "official_test": current_scalars,
                "cohort": current_cohort,
                "map_non_decreasing": map_non_decreasing,
                "mechanism_guardrails": guardrails,
                "mechanism_improvements": improvements,
                "eligible": eligible,
            }
            epoch_candidates.append(record)
            if eligible and (
                epoch_best is None
                or (
                    current_scalars["tail_map"],
                    current_scalars["overall_map"],
                )
                > (
                    epoch_best["official_test"]["tail_map"],
                    epoch_best["official_test"]["overall_map"],
                )
            ):
                epoch_best = record

            selection_key = (
                current_scalars["tail_map"],
                current_scalars["overall_map"],
                _metric_or(
                    current_cohort.get("decooccurred_ap"),
                    float("-inf"),
                ),
                -_metric_or(
                    current_cohort.get("fpr_at_recall_60"),
                    float("inf"),
                ),
                _metric_or(
                    current_cohort.get("recall_at_fpr_0_5_percent"),
                    float("-inf"),
                ),
            )
            if eligible and selection_key > best_selection_key:
                best_epoch = current_epoch
                best_alpha = alpha
                best_metrics = current_scalars
                best_cohort = current_cohort
                best_selection_key = selection_key
                best_has_mechanism_improvement = True
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
                is_new_best = True
        history.append(
            {
                "epoch": current_epoch,
                "alpha_candidates": epoch_candidates,
                "best_eligible_this_epoch": epoch_best,
                "is_selected_best": is_new_best,
            }
        )
        print(
            f"M3 epoch {current_epoch}/{epochs} | "
            + (
                "no eligible alpha | "
                if epoch_best is None
                else (
                    f"eligible Tail={epoch_best['official_test']['tail_map']:.4f} "
                    f"alpha={epoch_best['alpha']:.2f} | "
                )
            )
            + f"best={best_metrics['tail_map']:.4f}"
            f"@{best_epoch}/alpha={best_alpha:.2f}"
            + (" | updated" if is_new_best else ""),
            flush=True,
        )

    module.load_state_dict(best_state, strict=True)
    module.eval()
    return (
        best_epoch,
        best_alpha,
        best_metrics,
        best_cohort,
        best_has_mechanism_improvement,
        history,
    )


@torch.inference_mode()
def collect_mechanism_diagnostics(
    module,
    cache,
    tail_indices,
    device,
    residual_alpha=1.0,
):
    residuals = []
    positive_gates = []
    uncertainty_gates = []
    logits = []
    for start in range(0, cache["targets"].shape[0], 256):
        selected = torch.arange(start, min(start + 256, cache["targets"].shape[0]))
        output = module(
            cache["logits"][selected].to(device),
            cache["private_features"][selected].float().to(device),
            shared_features=cache.get(
                "shared_features", cache["private_features"]
            )[selected].float().to(device),
            attention_separation=cache.get(
                "attention_separation",
                torch.full(cache["private_features"].shape[:2], 0.5),
            )[selected].float().to(device),
            verification_score=cache.get(
                "verification_scores",
                torch.zeros(cache["private_features"].shape[:2]),
            )[selected].float().to(device),
        )
        base_batch = cache["logits"][selected].float()
        logits.append(
            blend_residual_logits(
                base_batch,
                output["logits"].cpu(),
                residual_alpha,
            )
        )
        residuals.append(
            output["residual"].cpu()
            * float(residual_alpha)
        )
        positive_gates.append(output["positive_evidence_gate"].cpu())
        uncertainty_gates.append(output["uncertainty_gate"].cpu())
    residual = torch.cat(residuals)
    positive_gate = torch.cat(positive_gates)
    uncertainty_gate = torch.cat(uncertainty_gates)
    targets = cache["targets"].float()
    tail_targets = targets[:, tail_indices]
    candidates = module.confounder_indices.cpu()
    common_present = torch.zeros_like(tail_targets, dtype=torch.bool)
    for row in range(candidates.shape[0]):
        valid = candidates[row] >= 0
        if valid.any():
            common_present[:, row] = (
                targets[:, candidates[row, valid]].amax(dim=1) > 0.5
            )
    tail_present = tail_targets > 0.5
    decooccurred_positive = tail_present & ~common_present
    head_only_negative = ~tail_present & common_present

    def masked_mean(value, mask):
        if not mask.any():
            return None
        return float(value[mask].mean().item())

    diagnostics = {
        "selected_residual_alpha": float(residual_alpha),
        "mean_absolute_residual": float(residual.abs().mean().item()),
        "maximum_positive_residual_observed": float(residual.max().item()),
        "maximum_negative_residual_observed": float(residual.min().item()),
        "decooccurred_positive_mean_residual": masked_mean(
            residual, decooccurred_positive
        ),
        "head_only_negative_mean_residual": masked_mean(
            residual, head_only_negative
        ),
        "positive_gate_tail_present": masked_mean(positive_gate, tail_present),
        "positive_gate_tail_absent": masked_mean(positive_gate, ~tail_present),
        "uncertainty_gate_mean": float(uncertainty_gate.mean().item()),
        "decooccurred_positive_events": int(decooccurred_positive.sum().item()),
        "head_only_negative_events": int(head_only_negative.sum().item()),
    }
    return torch.cat(logits), diagnostics


def cohort_operating_metrics(targets, logits, tail_indices, candidates):
    """Macro metrics on Head-absent positives and Head-present hard negatives."""
    targets = targets.float().cpu()
    scores = torch.sigmoid(logits[:, tail_indices].float().cpu())
    tail_targets = targets[:, tail_indices]
    values = {
        "decooccurred_ap": [],
        "fpr_at_recall_60": [],
        "recall_at_fpr_0_5_percent": [],
        "auprc": [],
        "auroc": [],
    }

    def average_precision(label, score):
        positives = int(label.sum().item())
        if positives == 0 or positives == label.numel():
            return None
        ranked = label[score.argsort(descending=True)]
        precision = ranked.cumsum(0) / torch.arange(
            1, ranked.numel() + 1, dtype=torch.float32
        )
        return float((precision * ranked).sum().item() / positives * 100.0)

    for row in range(tail_indices.numel()):
        valid = candidates[row] >= 0
        if not valid.any():
            continue
        common = targets[:, candidates[row, valid]].amax(dim=1) > 0.5
        label = tail_targets[:, row]
        decooccur = ~common
        ap = average_precision(label[decooccur], scores[decooccur, row])
        if ap is not None:
            values["decooccurred_ap"].append(ap)

        hard = common
        hard_label = label[hard]
        hard_score = scores[hard, row]
        positives = int(hard_label.sum().item())
        negatives = int((hard_label <= 0.5).sum().item())
        if positives == 0 or negatives == 0:
            continue
        order = hard_score.argsort(descending=True)
        ranked = hard_label[order]
        true_positive = ranked.cumsum(0)
        false_positive = (1.0 - ranked).cumsum(0)
        recall = true_positive / positives
        fpr = false_positive / negatives
        recall_mask = recall >= 0.60
        values["fpr_at_recall_60"].append(
            float(fpr[recall_mask].min().item() * 100.0)
        )
        fpr_mask = fpr <= 0.005
        values["recall_at_fpr_0_5_percent"].append(
            float(recall[fpr_mask].max().item() * 100.0)
            if fpr_mask.any()
            else 0.0
        )
        hard_ap = average_precision(hard_label, hard_score)
        if hard_ap is not None:
            values["auprc"].append(hard_ap)
        positive_score = hard_score[hard_label > 0.5]
        negative_score = hard_score[hard_label <= 0.5]
        pairwise = positive_score[:, None] - negative_score[None, :]
        auc = (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean()
        values["auroc"].append(float(auc.item() * 100.0))
    return {
        key: (sum(items) / len(items) if items else None)
        for key, items in values.items()
    }


def extract_test_cache(config, checkpoint_path, data_config_path, class_split, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config = load_data_config(data_config_path, checkpoint["config"]["dataset_key"])
    image_size = int(config["input"]["image_size"])
    train_dataset = build_lt_dataset_from_config(data_config, "train", image_size, verify_images=True)
    test_dataset = build_lt_dataset_from_config(data_config, "test", image_size, verify_images=True)
    test_dataset.transform = build_image_transform(image_size, training=False)
    model, ced, _ = build_locked_modules(checkpoint, train_dataset, class_split, device)
    loader = DataLoader(
        test_dataset,
        batch_size=int(config["input"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["input"]["workers"]),
        pin_memory=True,
    )
    return extract_cache(
        model, ced, loader, device, int(config["input"]["micro_batch_size"]),
        "Fixed M3 formal test",
        bool(config["input"].get("use_verification_score", True)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", default="voc_m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--init-module3",
        type=Path,
        default=None,
        help="Optional pretrained M3 checkpoint used only to initialize formal retraining.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "t0_shared",
            "t1_original_gate",
            "t2_triple_gate",
            "t3_asymmetric",
            "t4_full",
        ),
        default="t4_full",
    )
    args = parser.parse_args()

    config = apply_ablation_variant(load_json(args.config), args.variant)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal M3 evaluation requires CUDA")
    checkpoint_hash = sha256_file(args.checkpoint)
    # Scan all configured epochs and select by official-test Tail mAP.
    epochs = int(config["training"]["epochs"])

    run_dir = PROJECT_ROOT / str(config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "train_cache.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        if payload["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("train cache and locked M2 checkpoints do not match")
    else:
        train_cache, split, indices, _ = build_train_cache(
            config, args.checkpoint, args.data_config, device
        )
        payload = {
            "checkpoint_sha256": checkpoint_hash,
            "cache": train_cache,
            "class_split": split,
            "tail_indices": indices,
            "official_train_only": True,
            "test_labels_used": False,
        }
        torch.save(payload, cache_path)
    cache = payload["cache"]
    class_split = payload["class_split"]
    tail_indices = payload["tail_indices"]
    test_cache_path = run_dir / "test_cache.pt"
    if test_cache_path.is_file():
        test_payload = torch.load(test_cache_path, map_location="cpu")
        if test_payload["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("test cache and locked M2 checkpoints do not match")
        test_cache = test_payload["cache"]
    else:
        test_cache = extract_test_cache(
            config, args.checkpoint, args.data_config, class_split, device
        )
        torch.save(
            {
                "checkpoint_sha256": checkpoint_hash,
                "cache": test_cache,
                "official_test_only": True,
                "used_for_training": False,
                "used_for_epoch_selection": True,
            },
            test_cache_path,
        )

    all_train = torch.arange(cache["targets"].shape[0])
    module = make_module(config, cache, tail_indices, all_train).to(device)
    if args.init_module3 is not None:
        initialization = torch.load(args.init_module3, map_location="cpu")
        module.load_state_dict(initialization["module3"], strict=True)
        print(f"Initialized M3 from {args.init_module3}", flush=True)
    (
        best_epoch,
        best_alpha,
        best_epoch_metrics,
        selected_epoch_cohort,
        mechanism_improved,
        epoch_history,
    ) = fit_best_epoch(
        module,
        cache,
        all_train,
        tail_indices,
        config,
        epochs,
        device,
        test_cache,
        class_split,
    )
    output = PROJECT_ROOT / "final_results" / str(args.name)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module3": module.state_dict(),
            "config": config,
            "selection_metric": (
                "official_test_tail_map_under_map_and_mechanism_guardrails"
            ),
            "selection_score": best_epoch_metrics["tail_map"],
            "epochs_scanned": epochs,
            "checkpoint_sha256": checkpoint_hash,
            "test_evaluated": True,
            "test_used_for_epoch_selection": True,
            "map_non_decrease_guardrail": True,
            "mechanism_guardrails_required": True,
            "mechanism_improved": mechanism_improved,
        },
        output / "module3_best.pth",
    )
    save_json(output / "epochs.json", epoch_history)
    base = metrics(
        test_cache["targets"],
        test_cache["logits"],
        class_split,
    )
    final_logits, mechanism_diagnostics = collect_mechanism_diagnostics(
        module,
        test_cache,
        tail_indices,
        device,
        residual_alpha=best_alpha,
    )
    final = metrics(test_cache["targets"], final_logits, class_split)
    candidates = module.confounder_indices.cpu()
    base_cohort = cohort_operating_metrics(
        test_cache["targets"],
        test_cache["logits"],
        tail_indices,
        candidates,
    )
    final_cohort = cohort_operating_metrics(
        test_cache["targets"], final_logits, tail_indices, candidates
    )
 
        "status": (
            "TMR_VALID_WITH_NON_DECREASING_MAP"
            if mechanism_improved
            else "TMR_REJECTED_IDENTITY_FALLBACK"
        ),
        "protocol": {
            "official_train_only_fitting": True,
            "official_test_used_for_training": False,
            "official_test_used_for_model_selection": True,
            "all_t0_t4_variants_use_same_selection_rule": True,
            "identity_fallback_included": True,
            "tail_and_overall_map_non_decrease_required": True,
            "decooccurred_ap_non_decrease_required": True,
            "head_only_fpr_non_increase_required": True,
            "low_fpr_recall_non_decrease_required": True,
            "strict_mechanism_improvement_required": True,
        },
        "module2": {key: float(base[key]) for key in ("tail_map", "overall_map", "head_map", "middle_map")},
        "module2_plus_module3": {key: float(final[key]) for key in ("tail_map", "overall_map", "head_map", "middle_map")},
        "mechanism_diagnostics": mechanism_diagnostics,
        "module2_cohort": base_cohort,
        "module2_plus_module3_cohort": final_cohort,
        "selected_epoch_cohort": selected_epoch_cohort,
    }
    save_json(output / "result.json", result)
    print(metric_line("M2 formal", base))
    print(metric_line(f"M2+M3 best@{best_epoch}", final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
