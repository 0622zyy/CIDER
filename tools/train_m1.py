#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    build_lt_dataset_from_config,
    load_data_config,
)
from src.losses import AsymmetricLoss  # noqa: E402
from src.methods import (  # noqa: E402
    LCI,
)
from src.models import (  # noqa: E402
    build_cider_base,
)
from src.utils import (  # noqa: E402
    evaluate_multilabel_map,
    load_class_split,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = (
        torch.initial_seed() % (2**32)
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def split_micro_batches(
    images: Tensor,
    targets: Tensor,
    micro_batch_size: int,
):
    for start in range(
        0,
        images.shape[0],
        micro_batch_size,
    ):
        end = min(
            start + micro_batch_size,
            images.shape[0],
        )

        yield (
            images[start:end],
            targets[start:end],
        )


def build_optimizer_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
):
    decay = []
    no_decay = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim == 1
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return [
        {
            "params": no_decay,
            "lr": learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": decay,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
    ]


def intervention_scale(
    epoch_index: int,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    if epoch_index < start_epoch:
        return 0.0

    if ramp_epochs <= 0:
        return 1.0

    progress = (
        epoch_index - start_epoch + 1
    ) / ramp_epochs

    return float(
        min(
            max(progress, 0.0),
            1.0,
        )
    )


def make_train_loader(
    dataset,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        ),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def make_test_loader(
    dataset,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        ),
        worker_init_fn=worker_init_fn,
    )


def train_one_epoch(
    model: nn.Module,
    lci: Optional[nn.Module],
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    micro_batch_size: int,
    gradient_clip_norm: float,
    epoch_index: int,
    total_epochs: int,
    lci_scale: float,
    run_name: str,
) -> Dict[str, float]:
    model.train()

    totals = {
        "samples": 0,
        "classification_loss": 0.0,
        "auxiliary_loss": 0.0,
        "swap_positive_loss": 0.0,
        "swap_invariance_loss": 0.0,
        "core_necessity_loss": 0.0,
        "core_drop": 0.0,
        "cap": 0.0,
        "intervention_micro_batches": 0,
        "valid_donor_count": 0,
        "skipped_donor_count": 0,
    }

    progress = tqdm(
        loader,
        total=len(loader),
        desc=(
            f"{run_name} "
            f"{epoch_index + 1:03d}/"
            f"{total_epochs:03d}"
        ),
        dynamic_ncols=True,
        leave=True,
        mininterval=0.2,
        file=sys.stdout,
    )

    for images, targets in progress:
        full_batch_size = images.shape[0]

        optimizer.zero_grad(
            set_to_none=True
        )

        for (
            micro_images,
            micro_targets,
        ) in split_micro_batches(
            images,
            targets,
            micro_batch_size,
        ):
            micro_size = micro_images.shape[0]

            fraction = (
                micro_size
                / full_batch_size
            )

            micro_images = micro_images.to(
                device,
                non_blocking=True,
            )

            micro_targets = micro_targets.to(
                device,
                non_blocking=True,
            )

            with autocast(
                enabled=True
            ):
                if (
                    lci is not None
                    and lci_scale > 0
                ):
                    feature_map = (
                        model.forward_features(
                            micro_images
                        )
                    )

                    logits_for_loss = (
                        model.forward_head(
                            feature_map
                        )
                    )

                    classification_loss = (
                        criterion(
                            logits=logits_for_loss,
                            targets=micro_targets,
                        )
                    )

                    # LCI 内部需要再次使用原始 attention，
                    # 因此按统一接口重新完成干预前向。
                    result = lci(
                        model=model,
                        images=micro_images,
                        targets=micro_targets,
                        classification_loss=(
                            classification_loss
                        ),
                        intervention_scale=(
                            lci_scale
                        ),
                        precomputed_feature=(
                            feature_map
                        ),
                        precomputed_logits=(
                            logits_for_loss
                        ),
                    )

                    classification_loss = (
                        criterion(
                            logits=result["logits"],
                            targets=micro_targets,
                        )
                    )

                    auxiliary_loss = (
                        result[
                            "auxiliary_loss"
                        ]
                    )
                else:
                    result = None

                    logits = model(
                        micro_images
                    )

                    classification_loss = (
                        criterion(
                            logits=logits,
                            targets=micro_targets,
                        )
                    )

                    auxiliary_loss = (
                        classification_loss
                        * 0.0
                    )

                total_loss = (
                    classification_loss
                    + auxiliary_loss
                )

                weighted_loss = (
                    total_loss * fraction
                )

            if not torch.isfinite(
                total_loss
            ):
                raise RuntimeError(
                    "训练损失出现 NaN 或 Inf"
                )

            scaler.scale(
                weighted_loss
            ).backward()

            totals[
                "classification_loss"
            ] += (
                float(
                    classification_loss
                    .detach()
                    .item()
                )
                * micro_size
            )

            totals[
                "auxiliary_loss"
            ] += (
                float(
                    auxiliary_loss
                    .detach()
                    .item()
                )
                * micro_size
            )

            if result is not None:
                totals[
                    "swap_positive_loss"
                ] += (
                    float(
                        result[
                            "swap_positive_loss"
                        ]
                        .detach()
                        .item()
                    )
                    * micro_size
                )

                totals[
                    "swap_invariance_loss"
                ] += (
                    float(
                        result[
                            "swap_invariance_loss"
                        ]
                        .detach()
                        .item()
                    )
                    * micro_size
                )

                totals[
                    "core_necessity_loss"
                ] += (
                    float(
                        result[
                            "core_necessity_loss"
                        ]
                        .detach()
                        .item()
                    )
                    * micro_size
                )

                totals["core_drop"] += (
                    float(
                        result[
                            "mean_core_drop"
                        ]
                        .detach()
                        .item()
                    )
                    * micro_size
                )

                totals["cap"] += float(
                    result[
                        "cap_coefficient"
                    ]
                    .detach()
                    .item()
                )

                totals[
                    "intervention_micro_batches"
                ] += 1

                totals["valid_donor_count"] += int(
                    result["valid_donor_count"].detach().item()
                )
                totals["skipped_donor_count"] += int(
                    result["skipped_donor_count"].detach().item()
                )

            totals["samples"] += (
                micro_size
            )

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        scale_before = (
            scaler.get_scale()
        )

        scaler.step(optimizer)
        scaler.update()

        scale_after = (
            scaler.get_scale()
        )

        if scale_after < scale_before:
            tqdm.write(
                "[AMP] 检测到梯度溢出，"
                "当前优化步骤已跳过"
            )

        sample_count = max(
            totals["samples"],
            1,
        )

        progress.set_postfix(
            {
                "ASL": (
                    f"{totals['classification_loss'] / sample_count:.4f}"
                ),
                "LCI": (
                    f"{totals['auxiliary_loss'] / sample_count:.4f}"
                ),
                "scale": f"{lci_scale:.2f}",
            },
            refresh=False,
        )

    sample_count = max(
        totals["samples"],
        1,
    )

    intervention_count = max(
        totals[
            "intervention_micro_batches"
        ],
        1,
    )

    return {
        "classification_loss": (
            totals[
                "classification_loss"
            ]
            / sample_count
        ),
        "auxiliary_loss": (
            totals["auxiliary_loss"]
            / sample_count
        ),
        "swap_positive_loss": (
            totals[
                "swap_positive_loss"
            ]
            / sample_count
        ),
        "swap_invariance_loss": (
            totals[
                "swap_invariance_loss"
            ]
            / sample_count
        ),
        "core_necessity_loss": (
            totals[
                "core_necessity_loss"
            ]
            / sample_count
        ),
        "mean_core_drop": (
            totals["core_drop"]
            / sample_count
        ),
        "mean_cap": (
            totals["cap"]
            / intervention_count
        ),
        "lci_scale": float(
            lci_scale
        ),
        "valid_donor_count": int(
            totals["valid_donor_count"]
        ),
        "skipped_donor_count": int(
            totals["skipped_donor_count"]
        ),
        "valid_donor_rate": (
            totals["valid_donor_count"]
            / max(
                totals["valid_donor_count"]
                + totals["skipped_donor_count"],
                1,
            )
        ),
    }


@torch.inference_mode()
def evaluate_once(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    micro_batch_size: int,
    class_split: Dict[
        str,
        list[int],
    ],
    description: str,
) -> Dict[str, Any]:
    model.eval()

    score_batches = []
    target_batches = []

    progress = tqdm(
        loader,
        total=len(loader),
        desc=description,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.2,
        file=sys.stdout,
    )

    for images, targets in progress:
        current_scores = []

        for micro_images, _ in (
            split_micro_batches(
                images,
                targets,
                micro_batch_size,
            )
        ):
            micro_images = micro_images.to(
                device,
                non_blocking=True,
            )

            with autocast(
                enabled=True
            ):
                logits = model(
                    micro_images
                )

            current_scores.append(
                logits.float().cpu()
            )

        score_batches.append(
            torch.cat(
                current_scores,
                dim=0,
            )
        )

        target_batches.append(
            targets.float().cpu()
        )

    scores = torch.cat(
        score_batches,
        dim=0,
    )

    targets = torch.cat(
        target_batches,
        dim=0,
    )

    return evaluate_multilabel_map(
        targets=targets,
        scores=scores,
        class_split=class_split,
    )


def save_final_checkpoint(
    path: Path,
    model: nn.Module,
    config: Dict[str, Any],
    run_name: str,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        {
            "epoch": int(
                config["training"]["epochs"]
            ),
            "model": model.state_dict(),
            "config": config,
            "run_name": run_name,
            "selection": (
                "fixed_final_epoch"
            ),
            "ema": None,
        },
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def train_fixed_run(
    run_name: str,
    initial_state: Dict[str, Tensor],
    use_lci: bool,
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    train_dataset,
    device: torch.device,
    class_split: Dict[
        str,
        list[int],
    ],
    class_frequency: Tensor,
    output_dir: Path,
) -> nn.Module:
    seed = int(config["seed"])
    set_seed(seed)

    model_config = config["model"]
    training_config = config["training"]

    model = build_cider_base(
        num_classes=int(
            model_config["num_classes"]
        ),
        pretrained=False,
        num_groups=int(
            model_config["num_groups"]
        ),
        decoder_embedding=int(
            model_config[
                "decoder_embedding"
            ]
        ),
    )

    model.load_state_dict(
        initial_state,
        strict=True,
    )

    model = model.to(device)

    loss_config = config["loss"]

    criterion = AsymmetricLoss(
        gamma_negative=float(
            loss_config["gamma_negative"]
        ),
        gamma_positive=float(
            loss_config["gamma_positive"]
        ),
        clip=float(
            loss_config["clip"]
        ),
        reduction="mean",
    )

    optimizer = Adam(
        build_optimizer_groups(
            model=model,
            learning_rate=float(
                training_config[
                    "learning_rate"
                ]
            ),
            weight_decay=float(
                training_config[
                    "weight_decay"
                ]
            ),
        )
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(
            training_config["epochs"]
        ),
        eta_min=float(
            training_config[
                "minimum_learning_rate"
            ]
        ),
    )

    scaler = GradScaler(
        enabled=True
    )

    lci = None

    if use_lci:
        method_config = config["lci"]

        lci = (
            LCI(
                num_classes=int(
                    model_config[
                        "num_classes"
                    ]
                ),
                class_split=class_split,
                class_frequency=(
                    class_frequency
                ),
                core_topk_ratio=float(
                    method_config[
                        "core_topk_ratio"
                    ]
                ),
                necessity_margin=float(
                    method_config[
                        "necessity_margin"
                    ]
                ),
                swap_positive_weight=float(
                    method_config[
                        "swap_positive_weight"
                    ]
                ),
                swap_invariance_weight=float(
                    method_config[
                        "swap_invariance_weight"
                    ]
                ),
                core_necessity_weight=float(
                    method_config[
                        "core_necessity_weight"
                    ]
                ),
                auxiliary_to_classification_cap=float(
                    method_config[
                        "auxiliary_to_classification_cap"
                    ]
                ),
                head_sample_weight=float(
                    method_config[
                        "head_sample_weight"
                    ]
                ),
                middle_sample_weight=float(
                    method_config[
                        "middle_sample_weight"
                    ]
                ),
                tail_sample_weight=float(
                    method_config[
                        "tail_sample_weight"
                    ]
                ),
                mode=str(method_config.get("mode", "legacy")),
                soft_mask_temperature=float(
                    method_config.get("soft_mask_temperature", 0.5)
                ),
                ranking_margin=float(method_config.get("ranking_margin", 1.0)),
                ranking_weight=float(method_config.get("ranking_weight", 0.2)),
                maximum_frequency_weight=float(
                    method_config.get("maximum_frequency_weight", 3.0)
                ),
            ).to(device)
        )

    train_loader = make_train_loader(
        dataset=train_dataset,
        batch_size=int(
            training_config["batch_size"]
        ),
        workers=int(
            training_config["workers"]
        ),
        seed=seed,
    )

    metrics_path = (
        output_dir
        / f"{run_name.lower()}_train.jsonl"
    )

    for epoch_index in range(
        int(training_config["epochs"])
    ):
        if use_lci:
            method_config = config["lci"]

            current_lci_scale = (
                intervention_scale(
                    epoch_index=epoch_index,
                    start_epoch=int(
                        method_config[
                            "start_epoch"
                        ]
                    ),
                    ramp_epochs=int(
                        method_config[
                            "ramp_epochs"
                        ]
                    ),
                )
            )
        else:
            current_lci_scale = 0.0

        statistics = train_one_epoch(
            model=model,
            lci=lci,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            micro_batch_size=int(
                training_config[
                    "micro_batch_size"
                ]
            ),
            gradient_clip_norm=float(
                training_config[
                    "gradient_clip_norm"
                ]
            ),
            epoch_index=epoch_index,
            total_epochs=int(
                training_config["epochs"]
            ),
            lci_scale=(
                current_lci_scale
            ),
            run_name=run_name,
        )

        scheduler.step()

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    {
                        "epoch": (
                            epoch_index + 1
                        ),
                        "learning_rate": float(
                            optimizer.param_groups[
                                0
                            ]["lr"]
                        ),
                        "train": statistics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    save_final_checkpoint(
        path=(
            output_dir
            / f"{run_name.lower()}_final.pth"
        ),
        model=model,
        config=config,
        run_name=run_name,
    )

    return model


def metric_line(
    prefix: str,
    metrics: Dict[str, Any],
) -> str:
    return (
        f"{prefix:<16} | "
        f"Overall="
        f"{float(metrics['overall_map']):>7.4f} | "
        f"Head="
        f"{float(metrics['head_map']):>7.4f} | "
        f"Middle="
        f"{float(metrics['middle_map']):>7.4f} | "
        f"Tail="
        f"{float(metrics['tail_map']):>7.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config = load_json(args.config)

    output_dir = (
        Path(config["output_root"])
        / config["experiment_name"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "config_used.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    data_config = load_data_config(
        config_path=args.data_config,
        dataset_key=config[
            "dataset_key"
        ],
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "正式配对实验需要 CUDA"
        )

    torch.backends.cudnn.benchmark = True

    set_seed(int(config["seed"]))

    model_config = config["model"]

    # 只创建一次随机初始化，然后 Baseline 和 LCI 共用。
    initial_model = (
        build_cider_base(
            num_classes=int(
                model_config[
                    "num_classes"
                ]
            ),
            pretrained=bool(
                model_config[
                    "pretrained"
                ]
            ),
            num_groups=int(
                model_config["num_groups"]
            ),
            decoder_embedding=int(
                model_config[
                    "decoder_embedding"
                ]
            ),
        )
    )

    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in (
            initial_model.state_dict().items()
        )
    }

    del initial_model

    train_dataset = (
        build_lt_dataset_from_config(
            config=data_config,
            split="train",
            image_size=int(
                config["training"][
                    "image_size"
                ]
            ),
            return_metadata=False,
            verify_images=True,
        )
    )

    test_dataset = (
        build_lt_dataset_from_config(
            config=data_config,
            split="test",
            image_size=int(
                config["training"][
                    "image_size"
                ]
            ),
            return_metadata=False,
            verify_images=True,
        )
    )

    class_frequency = (
        train_dataset.class_counts.float()
    )

    class_split = load_class_split(
        path=data_config[
            "class_split"
        ],
        num_classes=int(
            model_config["num_classes"]
        ),
    )

    test_loader = make_test_loader(
        dataset=test_dataset,
        batch_size=int(
            config["training"][
                "batch_size"
            ]
        ),
        workers=int(
            config["training"][
                "workers"
            ]
        ),
    )

    print(
        f"Official {config['dataset_key']} paired experiment"
    )
    print(
        "Test evaluation: once after both "
        "fixed 40-epoch runs"
    )
    print()

    baseline_reference = config.get("baseline_reference_result")
    if baseline_reference:
        baseline_metrics = load_json(Path(baseline_reference))["baseline"]
    else:
        baseline_model = train_fixed_run(
            run_name="Baseline",
            initial_state=initial_state,
            use_lci=False,
            config=config,
            data_config=data_config,
            train_dataset=train_dataset,
            device=device,
            class_split=class_split,
            class_frequency=class_frequency,
            output_dir=output_dir,
        )
        baseline_metrics = evaluate_once(
            model=baseline_model,
            loader=test_loader,
            device=device,
            micro_batch_size=int(config["training"]["micro_batch_size"]),
            class_split=class_split,
            description="Official test: Baseline",
        )
        del baseline_model
        torch.cuda.empty_cache()

    lci_model = train_fixed_run(
        run_name="LCI",
        initial_state=initial_state,
        use_lci=True,
        config=config,
        data_config=data_config,
        train_dataset=train_dataset,
        device=device,
        class_split=class_split,
        class_frequency=(
            class_frequency
        ),
        output_dir=output_dir,
    )

    lci_metrics = evaluate_once(
        model=lci_model,
        loader=test_loader,
        device=device,
        micro_batch_size=int(
            config["training"][
                "micro_batch_size"
            ]
        ),
        class_split=class_split,
        description="Official test: LCI",
    )

    tail_gain = (
        float(lci_metrics["tail_map"])
        - float(
            baseline_metrics["tail_map"]
        )
    )

    overall_gain = (
        float(lci_metrics["overall_map"])
        - float(
            baseline_metrics[
                "overall_map"
            ]
        )
    )

    acceptance = config["acceptance"]

    if tail_gain >= float(
        acceptance["strong_candidate_tail_gain"]
    ):
        status = "STRONG_CANDIDATE"
    elif tail_gain >= float(
        acceptance[
            "minimum_candidate_tail_gain"
        ]
    ):
        status = "CANDIDATE"
    else:
        status = "REJECT_OR_REDESIGN"

    result = {
        "protocol": {
            "official_train_split": (
                data_config["train_list"]
            ),
            "official_test_split": (
                data_config["test_list"]
            ),
            "same_initialization": True,
            "same_training_order": True,
            "same_optimizer": True,
            "same_epochs": True,
            "test_evaluations_per_model": 1,
            "test_used_for_checkpoint_selection": False,
            "ema_used": False,
            "donor_policy": (
                "skip_anchor_when_no_target_negative_in_micro_batch"
            ),
            "invalid_donor_used": False,
        },
        "baseline": {
            "overall_map": float(
                baseline_metrics[
                    "overall_map"
                ]
            ),
            "head_map": float(
                baseline_metrics[
                    "head_map"
                ]
            ),
            "middle_map": float(
                baseline_metrics[
                    "middle_map"
                ]
            ),
            "tail_map": float(
                baseline_metrics[
                    "tail_map"
                ]
            )
        },
        "lci": {
            "overall_map": float(
                lci_metrics[
                    "overall_map"
                ]
            ),
            "head_map": float(
                lci_metrics["head_map"]
            ),
            "middle_map": float(
                lci_metrics[
                    "middle_map"
                ]
            ),
            "tail_map": float(
                lci_metrics["tail_map"]
            )
        },
        "gains": {
            "overall_map": overall_gain,
            "tail_map": tail_gain
        },
        "status": status
    }

    (
        output_dir / "result.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Official VOC-LT result")
    print("-" * 94)
    print(
        metric_line(
            "Baseline",
            baseline_metrics,
        )
    )
    print(
        metric_line(
            "Baseline + LCI",
            lci_metrics,
        )
    )
    print("-" * 94)
    print(
        f"Overall gain : "
        f"{overall_gain:+.4f}"
    )
    print(
        f"Tail gain    : "
        f"{tail_gain:+.4f}"
    )
    print(
        f"Status       : {status}"
    )
    print(
        "Test tuning  : False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
