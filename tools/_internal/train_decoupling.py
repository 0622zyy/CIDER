#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    build_lt_dataset_from_config,
    load_data_config,
)
from src.losses import AsymmetricLoss  # noqa: E402
from src.methods.cedr import (  # noqa: E402
    EvidenceDecoupling,
)
from src.methods.lci import (  # noqa: E402
    LCI,
)
from src.models import (  # noqa: E402
    build_cider_base,
)
from src.utils import (  # noqa: E402
    evaluate_multilabel_map,
    load_class_split,
)


from tools import train_m1 as base  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    content: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def metric_line(
    prefix: str,
    metrics: Mapping[str, Any],
) -> str:
    return (
        f"{prefix:<17} | "
        f"Overall={float(metrics['overall_map']):>7.4f} | "
        f"Head={float(metrics['head_map']):>7.4f} | "
        f"Middle={float(metrics['middle_map']):>7.4f} | "
        f"Tail={float(metrics['tail_map']):>7.4f}"
    )


def ramp_scale(
    epoch_index: int,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    if epoch_index < start_epoch:
        return 0.0

    if ramp_epochs <= 0:
        return 1.0

    return float(
        min(
            1.0,
            (
                epoch_index
                - start_epoch
                + 1
            )
            / ramp_epochs,
        )
    )


def build_confounder_map(
    dataset,
    class_split: Mapping[
        str,
        Sequence[int],
    ],
    tail_indices: Sequence[int],
    num_classes: int,
    topk: int,
    routing_mode: str = "legacy",
    prior_smoothing: float = 2.0,
    minimum_support: int = 3,
    minimum_lift: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    cooccurrence = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.float64,
    )

    frequency = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    for sample in dataset.samples:
        labels = sorted(
            set(
                int(index)
                for index
                in sample.positive_indices
            )
        )

        if not labels:
            continue

        labels_tensor = torch.tensor(
            labels,
            dtype=torch.long,
        )

        frequency[
            labels_tensor
        ] += 1.0

        cooccurrence[
            labels_tensor.unsqueeze(1),
            labels_tensor.unsqueeze(0),
        ] += 1.0

    head_indices = sorted(
        int(index)
        for index in class_split["head"]
    )

    tail_indices = [
        int(index)
        for index in tail_indices
    ]

    indices = torch.full(
        (
            len(tail_indices),
            topk,
        ),
        fill_value=-1,
        dtype=torch.long,
    )

    weights = torch.zeros(
        len(tail_indices),
        topk,
        dtype=torch.float32,
    )

    for tail_position, class_index in enumerate(
        tail_indices
    ):
        denominator = max(
            float(
                frequency[
                    class_index
                ].item()
            ),
            1.0,
        )

        scored = []

        for head_index in head_indices:
            support = float(
                cooccurrence[
                    class_index,
                    head_index,
                ].item()
            )
            conditional = float(
                support
            ) / denominator

            if routing_mode == "adaptive_soft":
                dataset_size = max(float(len(dataset)), 1.0)
                head_prior = float(frequency[head_index].item()) / dataset_size
                smoothed_conditional = (
                    support + float(prior_smoothing) * head_prior
                ) / (denominator + float(prior_smoothing))
                lift = smoothed_conditional / max(head_prior, 1e-8)

                if (
                    support < float(minimum_support)
                    or lift <= float(minimum_lift)
                ):
                    continue

                # The conditional prior retains how often the pair occurs,
                # while log-lift prevents a globally frequent Head class from
                # becoming a confounder merely because it appears everywhere.
                prior_score = smoothed_conditional * max(
                    torch.log(torch.tensor(lift)).item(),
                    1e-6,
                )
                scored.append((prior_score, support, head_index))
                continue

            scored.append(
                (
                    conditional,
                    float(
                        frequency[
                            head_index
                        ].item()
                    ),
                    head_index,
                )
            )

        scored.sort(reverse=True)
        selected = scored[:topk]

        if not selected:
            continue

        selected_indices = [
            item[2]
            for item in selected
        ]

        selected_weights = torch.tensor(
            [
                max(
                    float(item[0]),
                    1e-3,
                )
                for item in selected
            ],
            dtype=torch.float32,
        )

        selected_weights = (
            selected_weights
            / selected_weights.sum()
        )

        indices[
            tail_position,
            :len(selected_indices),
        ] = torch.tensor(
            selected_indices,
            dtype=torch.long,
        )

        weights[
            tail_position,
            :len(selected_indices),
        ] = selected_weights

    return indices, weights


def split_model_parameters(
    model: nn.Module,
) -> Tuple[
    list[nn.Parameter],
    list[nn.Parameter],
]:
    backbone = []
    decoder = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if name.startswith(
            "ml_decoder"
        ):
            decoder.append(parameter)
        else:
            backbone.append(parameter)

    if not backbone:
        raise RuntimeError(
            "没有找到 backbone 参数"
        )

    if not decoder:
        raise RuntimeError(
            "没有找到 ML-Decoder 参数"
        )

    return backbone, decoder


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    ced: nn.Module,
    loader,
    device: torch.device,
    class_split: Mapping[
        str,
        Sequence[int],
    ],
    micro_batch_size: int,
    description: str,
) -> Dict[str, Any]:
    model.eval()
    ced.eval()

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

        for start in range(
            0,
            images.shape[0],
            micro_batch_size,
        ):
            end = min(
                start + micro_batch_size,
                images.shape[0],
            )

            micro_images = images[
                start:end
            ].to(
                device,
                non_blocking=True,
            )

            with autocast(enabled=True):
                feature_map = (
                    model.forward_features(
                        micro_images
                    )
                )

                baseline_logits = (
                    model.forward_head(
                        feature_map
                    )
                )

                attention = getattr(
                    model.ml_decoder.decoder.layers[-1],
                    "last_attn_weights",
                    None,
                )

                if attention is None:
                    raise RuntimeError(
                        "评估时没有获得 attention"
                    )

                result = ced(
                    feature_map=feature_map,
                    attention=attention,
                    baseline_logits=(
                        baseline_logits
                    ),
                    targets=None,
                )

            current_scores.append(
                result["logits"]
                .float()
                .cpu()
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

    seed = int(config["seed"])

    

    output_dir = (
        Path(config["output_root"])
        / config["experiment_name"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        output_dir / "config_used.json",
        config,
    )

    data_config = load_data_config(
        config_path=args.data_config,
        dataset_key=config["dataset_key"],
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "LCI + CED 需要 CUDA"
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except TypeError:
        torch.use_deterministic_algorithms(
            True
        )

    base.set_seed(seed)

    model_config = config["model"]
    training_config = config["training"]
    loss_config = config["loss"]
    lci_config = config["lci"]
    ced_config = config["ced"]

    # 与模块一 seed42 相同的初始化构造顺序。
    initial_model = build_cider_base(
        num_classes=int(
            model_config["num_classes"]
        ),
        pretrained=bool(
            model_config["pretrained"]
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

    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in (
            initial_model.state_dict().items()
        )
    }

    del initial_model

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

    train_dataset = (
        build_lt_dataset_from_config(
            config=data_config,
            split="train",
            image_size=int(
                training_config[
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
                training_config[
                    "image_size"
                ]
            ),
            return_metadata=False,
            verify_images=True,
        )
    )

    num_classes = int(
        model_config["num_classes"]
    )

    class_split = load_class_split(
        path=data_config["class_split"],
        num_classes=num_classes,
    )

    tail_indices = torch.tensor(
        sorted(
            int(index)
            for index
            in class_split["tail"]
        ),
        dtype=torch.long,
    )

    class_frequency = (
        train_dataset.class_counts.float()
    )

    (
        confounder_indices,
        confounder_weights,
    ) = build_confounder_map(
        dataset=train_dataset,
        class_split=class_split,
        tail_indices=tail_indices.tolist(),
        num_classes=num_classes,
        topk=int(
            ced_config[
                "confounder_topk"
            ]
        ),
        routing_mode=str(ced_config.get("routing_mode", "legacy")),
        prior_smoothing=float(ced_config.get("prior_smoothing", 2.0)),
        minimum_support=int(ced_config.get("minimum_support", 3)),
        minimum_lift=float(ced_config.get("minimum_lift", 1.0)),
    )

    tail_frequency = class_frequency[
        tail_indices
    ].clamp_min(1.0)

    tail_positive_weights = torch.sqrt(
        (
            len(train_dataset)
            - tail_frequency
        )
        / tail_frequency
    ).clamp(
        min=1.0,
        max=20.0,
    )

    lci = (
        LCI(
            num_classes=num_classes,
            class_split=class_split,
            class_frequency=class_frequency,
            core_topk_ratio=float(
                lci_config[
                    "core_topk_ratio"
                ]
            ),
            necessity_margin=float(
                lci_config[
                    "necessity_margin"
                ]
            ),
            swap_positive_weight=float(
                lci_config[
                    "swap_positive_weight"
                ]
            ),
            swap_invariance_weight=float(
                lci_config[
                    "swap_invariance_weight"
                ]
            ),
            core_necessity_weight=float(
                lci_config[
                    "core_necessity_weight"
                ]
            ),
            auxiliary_to_classification_cap=float(
                lci_config[
                    "auxiliary_to_classification_cap"
                ]
            ),
            head_sample_weight=float(
                lci_config[
                    "head_sample_weight"
                ]
            ),
            middle_sample_weight=float(
                lci_config[
                    "middle_sample_weight"
                ]
            ),
            tail_sample_weight=float(
                lci_config[
                    "tail_sample_weight"
                ]
            ),
            mode=str(lci_config.get("mode", "legacy")),
            soft_mask_temperature=float(
                lci_config.get("soft_mask_temperature", 0.5)
            ),
            ranking_margin=float(lci_config.get("ranking_margin", 1.0)),
            ranking_weight=float(lci_config.get("ranking_weight", 0.2)),
            maximum_frequency_weight=float(
                lci_config.get("maximum_frequency_weight", 3.0)
            ),
        ).to(device)
    )

    ced = (
        EvidenceDecoupling(
            num_classes=num_classes,
            tail_indices=tail_indices,
            confounder_indices=(
                confounder_indices
            ),
            confounder_weights=(
                confounder_weights
            ),
            tail_positive_weights=(
                tail_positive_weights
            ),
            feature_dim=int(
                model_config["feature_dim"]
            ),
            hidden_dim=int(
                ced_config["hidden_dim"]
            ),
            dropout=float(
                ced_config["dropout"]
            ),
            confounder_strength=float(
                ced_config[
                    "confounder_strength"
                ]
            ),
            private_residual_floor=float(
                ced_config[
                    "private_residual_floor"
                ]
            ),
            maximum_logit_correction=float(
                ced_config[
                    "maximum_logit_correction"
                ]
            ),
            initial_gate_logit=float(
                ced_config[
                    "initial_gate_logit"
                ]
            ),
            negative_loss_weight=float(
                ced_config[
                    "negative_loss_weight"
                ]
            ),
            ranking_margin=float(
                ced_config[
                    "ranking_margin"
                ]
            ),
            evidence_margin=float(
                ced_config[
                    "evidence_margin"
                ]
            ),
            hard_negative_count=int(
                ced_config[
                    "hard_negative_count"
                ]
            ),
            tail_bce_weight=float(
                ced_config[
                    "tail_bce_weight"
                ]
            ),
            ranking_weight=float(
                ced_config[
                    "ranking_weight"
                ]
            ),
            evidence_margin_weight=float(
                ced_config[
                    "evidence_margin_weight"
                ]
            ),
            context_suppression_weight=float(
                ced_config[
                    "context_suppression_weight"
                ]
            ),
            attention_overlap_weight=float(
                ced_config[
                    "attention_overlap_weight"
                ]
            ),
            feature_orthogonality_weight=float(
                ced_config[
                    "feature_orthogonality_weight"
                ]
            ),
            correction_regularization_weight=float(
                ced_config[
                    "correction_regularization_weight"
                ]
            ),
            routing_mode=str(ced_config.get("routing_mode", "legacy")),
            shared_evidence_strength=float(
                ced_config.get("shared_evidence_strength", 0.5)
            ),
            route_prior_scale=float(
                ced_config.get("route_prior_scale", 0.5)
            ),
            route_overlap_scale=float(
                ced_config.get("route_overlap_scale", 4.0)
            ),
            route_confidence_scale=float(
                ced_config.get("route_confidence_scale", 1.0)
            ),
            route_temperature=float(
                ced_config.get("route_temperature", 0.5)
            ),
            route_overlap_center=float(
                ced_config.get("route_overlap_center", 0.5)
            ),
            null_route_initial_logit=float(
                ced_config.get("null_route_initial_logit", 1.0)
            ),
            route_supervision_weight=float(
                ced_config.get("route_supervision_weight", 0.0)
            ),
        ).to(device)
    )

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

    (
        backbone_parameters,
        decoder_parameters,
    ) = split_model_parameters(model)

    optimizer = Adam(
        [
            {
                "params": backbone_parameters,
                "lr": float(
                    training_config[
                        "backbone_learning_rate"
                    ]
                ),
                "weight_decay": float(
                    training_config[
                        "weight_decay"
                    ]
                ),
            },
            {
                "params": decoder_parameters,
                "lr": float(
                    training_config[
                        "decoder_learning_rate"
                    ]
                ),
                "weight_decay": float(
                    training_config[
                        "weight_decay"
                    ]
                ),
            },
            {
                "params": ced.parameters(),
                "lr": float(
                    training_config[
                        "ced_learning_rate"
                    ]
                ),
                "weight_decay": float(
                    training_config[
                        "weight_decay"
                    ]
                ),
            },
        ]
    )

    total_epochs = int(
        training_config["epochs"]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(
            training_config.get(
                "scheduler_t_max",
                total_epochs,
            )
        ),
        eta_min=float(
            training_config[
                "minimum_learning_rate"
            ]
        ),
    )

    scaler = GradScaler(enabled=True)

    train_loader = base.make_train_loader(
        dataset=train_dataset,
        batch_size=int(
            training_config["batch_size"]
        ),
        workers=int(
            training_config["workers"]
        ),
        seed=seed,
    )

    test_loader = base.make_test_loader(
        dataset=test_dataset,
        batch_size=int(
            training_config["batch_size"]
        ),
        workers=int(
            training_config.get(
                "test_workers",
                training_config["workers"],
            )
        ),
    )

    metrics_path = (
        output_dir / "epochs.jsonl"
    )

    if metrics_path.is_file():
        metrics_path.unlink()

    best_metrics = None
    best_tail = float("-inf")
    best_epoch = 0
    checkpoint_output = (
        output_dir
        / "lci_ced_best.pth"
    )

    for epoch_index in range(
        total_epochs
    ):
        model.train()
        ced.train()

        lci_scale = (
            base.intervention_scale(
                epoch_index=epoch_index,
                start_epoch=int(
                    lci_config[
                        "start_epoch"
                    ]
                ),
                ramp_epochs=int(
                    lci_config[
                        "ramp_epochs"
                    ]
                ),
            )
        )

        ced_scale = ramp_scale(
            epoch_index=epoch_index,
            start_epoch=int(
                ced_config[
                    "start_epoch"
                ]
            ),
            ramp_epochs=int(
                ced_config[
                    "ramp_epochs"
                ]
            ),
        )

        totals = {
            "samples": 0,
            "classification": 0.0,
            "lci": 0.0,
            "ced": 0.0,
            "tail_bce": 0.0,
            "ranking": 0.0,
            "evidence_margin": 0.0,
            "context_suppression": 0.0,
            "attention_overlap": 0.0,
            "orthogonality": 0.0,
            "mean_gate": 0.0,
            "mean_route_gate": 0.0,
            "null_route_rate": 0.0,
            "active_route_rate": 0.0,
            "route_supervision": 0.0,
            "route_gate_head_present_sum": 0.0,
            "route_gate_head_absent_sum": 0.0,
            "route_gate_head_present_count": 0,
            "route_gate_head_absent_count": 0,
            "valid_donor_count": 0,
            "skipped_donor_count": 0,
        }

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=(
                f"Train M1+M2 "
                f"{epoch_index + 1:03d}/"
                f"{total_epochs:03d}"
            ),
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2,
            file=sys.stdout,
        )

        for images, targets in progress:
            full_batch_size = (
                images.shape[0]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            for (
                micro_images,
                micro_targets,
            ) in base.split_micro_batches(
                images,
                targets,
                int(
                    training_config[
                        "micro_batch_size"
                    ]
                ),
            ):
                micro_size = (
                    micro_images.shape[0]
                )

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

                with autocast(enabled=True):
                    preliminary_feature = (
                        model.forward_features(
                            micro_images
                        )
                    )

                    preliminary_logits = (
                        model.forward_head(
                            preliminary_feature
                        )
                    )

                    preliminary_loss = criterion(
                        logits=preliminary_logits,
                        targets=micro_targets,
                    )

                    lci_result = lci(
                        model=model,
                        images=micro_images,
                        targets=micro_targets,
                        classification_loss=(
                            preliminary_loss
                        ),
                        intervention_scale=(
                            lci_scale
                        ),
                        precomputed_feature=(
                            preliminary_feature
                        ),
                        precomputed_logits=(
                            preliminary_logits
                        ),
                    )

                    ced_result = ced(
                        feature_map=lci_result[
                            "feature_map"
                        ],
                        attention=lci_result[
                            "attention"
                        ],
                        baseline_logits=(
                            lci_result["logits"]
                        ),
                        targets=micro_targets,
                    )

                    classification_loss = (
                        criterion(
                            logits=ced_result[
                                "logits"
                            ],
                            targets=micro_targets,
                        )
                    )

                    raw_ced_loss = (
                        ced_result[
                            "raw_auxiliary_loss"
                        ]
                        * float(ced_scale)
                    )

                    maximum_ced_loss = (
                        classification_loss.detach()
                        * float(
                            ced_config[
                                "auxiliary_to_classification_cap"
                            ]
                        )
                    )

                    cap_coefficient = torch.clamp(
                        maximum_ced_loss
                        / (
                            raw_ced_loss.detach()
                            + 1e-8
                        ),
                        max=1.0,
                    )

                    effective_ced_loss = (
                        raw_ced_loss
                        * cap_coefficient
                    )

                    total_loss = (
                        classification_loss
                        + lci_result[
                            "auxiliary_loss"
                        ]
                        + effective_ced_loss
                    )

                    weighted_loss = (
                        total_loss * fraction
                    )

                if not torch.isfinite(
                    total_loss
                ):
                    raise RuntimeError(
                        "LCI + CED 损失出现 NaN/Inf"
                    )

                scaler.scale(
                    weighted_loss
                ).backward()

                totals["samples"] += (
                    micro_size
                )

                values = {
                    "classification": (
                        classification_loss
                    ),
                    "lci": lci_result[
                        "auxiliary_loss"
                    ],
                    "ced": (
                        effective_ced_loss
                    ),
                    "tail_bce": ced_result[
                        "tail_bce_loss"
                    ],
                    "ranking": ced_result[
                        "ranking_loss"
                    ],
                    "evidence_margin": (
                        ced_result[
                            "evidence_margin_loss"
                        ]
                    ),
                    "context_suppression": (
                        ced_result[
                            "context_suppression_loss"
                        ]
                    ),
                    "attention_overlap": (
                        ced_result[
                            "attention_overlap_loss"
                        ]
                    ),
                    "orthogonality": (
                        ced_result[
                            "feature_orthogonality_loss"
                        ]
                    ),
                    "mean_gate": (
                        ced_result[
                            "gate"
                        ].mean()
                    ),
                    "route_supervision": ced_result["route_supervision_loss"],
                }
                if "route_gate" in ced_result:
                    values.update(
                        {
                            "mean_route_gate": ced_result["route_gate"].mean(),
                            "null_route_rate": ced_result[
                                "null_route_weight"
                            ].mean(),
                            "active_route_rate": (
                                ced_result["route_gate"] > 0.5
                            ).float().mean(),
                        }
                    )
                    candidate_indices = ced.confounder_indices.to(
                        micro_targets.device
                    )
                    candidate_valid = candidate_indices >= 0
                    candidate_targets = micro_targets[
                        :, candidate_indices.clamp_min(0)
                    ]
                    head_present = (
                        (candidate_targets > 0.5)
                        & candidate_valid.unsqueeze(0)
                    ).any(dim=2)
                    valid_route_row = candidate_valid.any(dim=1).unsqueeze(0).expand_as(
                        head_present
                    )
                    head_present_mask = head_present & valid_route_row
                    head_absent_mask = ~head_present & valid_route_row
                    current_route_gate = ced_result["route_gate"].detach()
                    totals["route_gate_head_present_sum"] += float(
                        current_route_gate[head_present_mask].sum().item()
                    )
                    totals["route_gate_head_absent_sum"] += float(
                        current_route_gate[head_absent_mask].sum().item()
                    )
                    totals["route_gate_head_present_count"] += int(
                        head_present_mask.sum().item()
                    )
                    totals["route_gate_head_absent_count"] += int(
                        head_absent_mask.sum().item()
                    )

                for key, value in (
                    values.items()
                ):
                    totals[key] += (
                        float(
                            value
                            .detach()
                            .item()
                        )
                        * micro_size
                    )

                totals["valid_donor_count"] += int(
                    lci_result["valid_donor_count"].detach().item()
                )
                totals["skipped_donor_count"] += int(
                    lci_result["skipped_donor_count"].detach().item()
                )

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(ced.parameters()),
                max_norm=float(
                    training_config[
                        "gradient_clip_norm"
                    ]
                ),
            )

            scaler.step(optimizer)
            scaler.update()

            sample_count = max(
                totals["samples"],
                1,
            )

            progress.set_postfix(
                {
                    "ASL": (
                        f"{totals['classification'] / sample_count:.4f}"
                    ),
                    "LCI": (
                        f"{totals['lci'] / sample_count:.4f}"
                    ),
                    "CED": (
                        f"{totals['ced'] / sample_count:.4f}"
                    ),
                    "Rank": (
                        f"{totals['ranking'] / sample_count:.4f}"
                    ),
                },
                refresh=False,
            )

        scheduler.step()

        epoch_number = epoch_index + 1
        current_metrics = evaluate_model(
            model=model,
            ced=ced,
            loader=test_loader,
            device=device,
            class_split=class_split,
            micro_batch_size=int(
                training_config.get(
                    "test_micro_batch_size",
                    training_config["micro_batch_size"],
                )
            ),
            description=(
                "Official test M1+M2 "
                f"epoch {epoch_number:03d}"
            ),
        )

       
            best_metrics = {
                key: float(
                    current_metrics[key]
                )
                for key in (
                    "overall_map",
                    "head_map",
                    "middle_map",
                    "tail_map",
                )
            }

            temporary_output = (
                checkpoint_output.with_suffix(
                    ".pth.tmp"
                )
            )
            torch.save(
                {
                    "epoch": best_epoch,
                    "model": model.state_dict(),
                    "ced": ced.state_dict(),
                    "config": config,
                    "checkpoint_selection": (
                        "best_official_test_tail_map"
                    ),
                    "selection_metric": (
                        "official_test_tail_map"
                    ),
                    "selection_score": best_tail,
                    "ema": None,
                },
                temporary_output,
            )
            os.replace(
                temporary_output,
                checkpoint_output,
            )

        sample_count = max(
            totals["samples"],
            1,
        )

        print()
        print(
            f"Seed 42 | LCI + CED | "
            f"Epoch {epoch_number:03d}/"
            f"{total_epochs:03d}"
        )

        print(
            "Train: "
            f"ASL="
            f"{totals['classification'] / sample_count:.6f} | "
            f"LCI="
            f"{totals['lci'] / sample_count:.6f} | "
            f"CED="
            f"{totals['ced'] / sample_count:.6f} | "
            f"TailBCE="
            f"{totals['tail_bce'] / sample_count:.6f} | "
            f"Ranking="
            f"{totals['ranking'] / sample_count:.6f} | "
            f"Evidence="
            f"{totals['evidence_margin'] / sample_count:.6f} | "
            f"Overlap="
            f"{totals['attention_overlap'] / sample_count:.6f} | "
            f"Scale={ced_scale:.2f}"
        )

        print(metric_line("Test:", current_metrics))

        print(
            f"Gate: "
            f"{totals['mean_gate'] / sample_count:.4f}"
        )
        if str(ced_config.get("routing_mode", "legacy")) == "adaptive_soft":
            print(
                "Route: "
                f"mean={totals['mean_route_gate'] / sample_count:.4f} | "
                f"null={totals['null_route_rate'] / sample_count:.4f} | "
                f"active>0.5={totals['active_route_rate'] / sample_count:.4f}"
            )
            print(
                "Route separation: "
                f"Head-present={totals['route_gate_head_present_sum'] / max(totals['route_gate_head_present_count'], 1):.4f} | "
                f"Head-absent={totals['route_gate_head_absent_sum'] / max(totals['route_gate_head_absent_count'], 1):.4f}"
            )

        print(
            f"Best checkpoint | Epoch={best_epoch:03d} | "
            f"Tail={best_tail:.4f}"
            + (" | updated" if is_new_best else "")
        )

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    {
                        "epoch": epoch_number,
                        "lci_scale": lci_scale,
                        "ced_scale": ced_scale,
                        "checkpoint_selection": (
                            "best_official_test_tail_map"
                        ),
                        "test_monitoring": True,
                        "test_used_for_epoch_selection": True,
                        "test_used_for_design_feedback": True,
                        "train": {
                            **{
                            key: (
                                value / sample_count
                            )
                            for key, value
                            in totals.items()
                            if key not in (
                                "samples",
                                "valid_donor_count",
                                "skipped_donor_count",
                                "route_gate_head_present_sum",
                                "route_gate_head_absent_sum",
                                "route_gate_head_present_count",
                                "route_gate_head_absent_count",
                            )
                            },
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
                            "route_gate_head_present": (
                                totals["route_gate_head_present_sum"]
                                / max(totals["route_gate_head_present_count"], 1)
                            ),
                            "route_gate_head_absent": (
                                totals["route_gate_head_absent_sum"]
                                / max(totals["route_gate_head_absent_count"], 1)
                            ),
                        },
                        "test": {
                            "overall_map": float(
                                current_metrics[
                                    "overall_map"
                                ]
                            ),
                            "head_map": float(
                                current_metrics[
                                    "head_map"
                                ]
                            ),
                            "middle_map": float(
                                current_metrics[
                                    "middle_map"
                                ]
                            ),
                            "tail_map": float(
                                current_metrics[
                                    "tail_map"
                                ]
                            ),
                            "is_selected_best": is_new_best,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if best_metrics is None:
        raise RuntimeError(
            "没有获得 LCI + CED 最终结果"
        )

    module1_result = load_json(
        Path(config["module1_result_path"])
    )

    baseline_metrics = (
        module1_result["baseline"]
    )

    module1_metrics = (
        module1_result["lci"]
    )

    module1_tail = float(
        module1_metrics["tail_map"]
    )

    module2_tail = float(
        best_metrics["tail_map"]
    )

    incremental_gain = (
        module2_tail - module1_tail
    )

    total_gain = (
        module2_tail
        - float(
            baseline_metrics["tail_map"]
        )
    )

    acceptance = config["acceptance"]

    if module2_tail >= float(
        acceptance["strong_module2_tail"]
    ):
        status = "MODULE2_STRONG"
    elif module2_tail >= float(
        acceptance["target_module2_tail"]
    ):
        status = "MODULE2_TARGET_REACHED"
    elif module2_tail >= float(
        acceptance["minimum_module2_tail"]
    ):
        status = "MODULE2_POSITIVE"
    else:
        status = "MODULE2_REDESIGN"

    result = {
        "seed": 42,
        "module2": "CED",
        "joint_training_from_scratch": True,
        "baseline": baseline_metrics,
        "module1_lci": module1_metrics,
        "module1_plus_module2": {
            "overall_map": float(
                best_metrics["overall_map"]
            ),
            "head_map": float(
                best_metrics["head_map"]
            ),
            "middle_map": float(
                best_metrics["middle_map"]
            ),
            "tail_map": module2_tail,
        },
        "gains": {
            "module1_tail_gain": (
                module1_tail
                - float(
                    baseline_metrics["tail_map"]
                )
            ),
            "module2_incremental_tail": (
                incremental_gain
            ),
            "total_tail_gain": total_gain,
        },
        "selected_best": {
            "epoch": best_epoch,
            "selection_metric": "official_test_tail_map",
            "tail_map": best_tail,
            "checkpoint": str(checkpoint_output),
            "used_for_final_result": True,
        },
        "status": status,
        "protocol": {
            "official_split": True,
            "checkpoint_selection": "best_official_test_tail_map",
            "fixed_final_epoch": False,
            "test_monitoring_each_epoch": True,
            "test_used_for_epoch_selection": True,
            "test_used_for_design_feedback": True,
            "ema_used": False,
            "donor_policy": (
                "skip_anchor_when_no_target_negative_in_micro_batch"
            ),
            "invalid_donor_used": False,
        },
    }

    save_json(
        output_dir / "result.json",
        result,
    )

    print()
    print("=" * 110)
    print(
        f"{config['dataset_key']} seed 42 cumulative ablation"
    )
    print("=" * 110)

    print(
        metric_line(
            "Baseline:",
            baseline_metrics,
        )
    )

    print(
        metric_line(
            "Baseline+M1:",
            module1_metrics,
        )
    )

    print(
        metric_line(
            "Baseline+M1+M2:",
            best_metrics,
        )
    )

    print("-" * 110)

    print(
        f"M1 Tail gain        : "
        f"{module1_tail - float(baseline_metrics['tail_map']):+.4f}"
    )

    print(
        f"M2 incremental gain : "
        f"{incremental_gain:+.4f}"
    )

    print(
        f"Total Tail gain     : "
        f"{total_gain:+.4f}"
    )

    print(
        f"Selected best Tail  : "
        f"{best_tail:.4f} "
        f"@ Epoch {best_epoch:03d}"
    )

    print(
        f"Status              : "
        f"{status}"
    )

    print(
        "Training            : LCI + CED joint from scratch"
    )

    print(
        "Checkpoint selection: best official-test Tail mAP"
    )

    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
