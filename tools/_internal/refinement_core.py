#!/usr/bin/env python3

"""Internal cache utilities for CEDR memory-refinement phase."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import build_lt_dataset_from_config, load_data_config
from src.data.lt_multilabel_dataset import build_image_transform
from src.methods.cedr import (
    EvidenceDecoupling,
    MemoryReliabilityRefinement,
)
from src.methods._internal.tmr import (
    build_tail_dependency_graph,
    build_tail_memories,
)
from src.models import build_cider_base
from src.utils import evaluate_multilabel_map, load_class_split
from tools._internal.train_decoupling import build_confounder_map

CED = EvidenceDecoupling
TMR = MemoryReliabilityRefinement


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_line(name: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"{name:<16} | Tail={float(metrics['tail_map']):>7.4f} | "
        f"Overall={float(metrics['overall_map']):>7.4f} | "
        f"Head={float(metrics['head_map']):>7.4f} | "
        f"Middle={float(metrics['middle_map']):>7.4f}"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_locked_modules(
    checkpoint: Mapping[str, Any],
    train_dataset,
    class_split: Mapping[str, Sequence[int]],
    device: torch.device,
) -> tuple[nn.Module, nn.Module, Tensor]:
    source = checkpoint["config"]
    model_config = source["model"]
    ced_config = source["ced"]
    num_classes = int(model_config["num_classes"])
    model = build_cider_base(
        num_classes=num_classes,
        pretrained=False,
        num_groups=int(model_config["num_groups"]),
        decoder_embedding=int(model_config["decoder_embedding"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    tail_indices = torch.tensor(sorted(class_split["tail"]), dtype=torch.long)
    confounder_indices, confounder_weights = build_confounder_map(
        dataset=train_dataset,
        class_split=class_split,
        tail_indices=tail_indices.tolist(),
        num_classes=num_classes,
        topk=int(ced_config["confounder_topk"]),
        routing_mode=str(ced_config.get("routing_mode", "legacy")),
        prior_smoothing=float(ced_config.get("prior_smoothing", 2.0)),
        minimum_support=int(ced_config.get("minimum_support", 3)),
        minimum_lift=float(ced_config.get("minimum_lift", 1.0)),
    )
    frequency = train_dataset.class_counts.float()[tail_indices].clamp_min(1.0)
    positive_weights = torch.sqrt(
        (len(train_dataset) - frequency) / frequency
    ).clamp(1.0, 20.0)
    ced = CED(
        num_classes=num_classes,
        tail_indices=tail_indices,
        confounder_indices=confounder_indices,
        confounder_weights=confounder_weights,
        tail_positive_weights=positive_weights,
        feature_dim=int(model_config["feature_dim"]),
        hidden_dim=int(ced_config["hidden_dim"]),
        dropout=float(ced_config["dropout"]),
        confounder_strength=float(ced_config["confounder_strength"]),
        private_residual_floor=float(ced_config["private_residual_floor"]),
        maximum_logit_correction=float(ced_config["maximum_logit_correction"]),
        initial_gate_logit=float(ced_config["initial_gate_logit"]),
        negative_loss_weight=float(ced_config["negative_loss_weight"]),
        ranking_margin=float(ced_config["ranking_margin"]),
        evidence_margin=float(ced_config["evidence_margin"]),
        hard_negative_count=int(ced_config["hard_negative_count"]),
        tail_bce_weight=float(ced_config["tail_bce_weight"]),
        ranking_weight=float(ced_config["ranking_weight"]),
        evidence_margin_weight=float(ced_config["evidence_margin_weight"]),
        context_suppression_weight=float(ced_config["context_suppression_weight"]),
        attention_overlap_weight=float(ced_config["attention_overlap_weight"]),
        feature_orthogonality_weight=float(ced_config["feature_orthogonality_weight"]),
        correction_regularization_weight=float(
            ced_config["correction_regularization_weight"]
        ),
        routing_mode=str(ced_config.get("routing_mode", "legacy")),
        shared_evidence_strength=float(
            ced_config.get("shared_evidence_strength", 0.5)
        ),
        route_prior_scale=float(ced_config.get("route_prior_scale", 0.5)),
        route_overlap_scale=float(ced_config.get("route_overlap_scale", 4.0)),
        route_confidence_scale=float(
            ced_config.get("route_confidence_scale", 1.0)
        ),
        route_temperature=float(ced_config.get("route_temperature", 0.5)),
        route_overlap_center=float(
            ced_config.get("route_overlap_center", 0.5)
        ),
        null_route_initial_logit=float(
            ced_config.get("null_route_initial_logit", 1.0)
        ),
        route_supervision_weight=float(
            ced_config.get("route_supervision_weight", 0.0)
        ),
    )
    ced.load_state_dict(checkpoint["ced"], strict=True)
    return model.to(device).eval(), ced.to(device).eval(), tail_indices


@torch.inference_mode()
def extract_cache(
    model: nn.Module,
    ced: nn.Module,
    loader: DataLoader,
    device: torch.device,
    micro_batch_size: int,
    description: str,
    use_verification_score: bool = True,
) -> Dict[str, Tensor]:
    logits_batches = []
    single_logits_batches = []
    private_batches = []
    shared_batches = []
    separation_batches = []
    verification_batches = []
    target_batches = []
    for images, targets in tqdm(
        loader, desc=description, dynamic_ncols=True, file=sys.stdout
    ):
        current_logits = []
        current_single_logits = []
        current_private = []
        current_shared = []
        current_separation = []
        current_verification = []
        for start in range(0, images.shape[0], micro_batch_size):
            micro = images[start : start + micro_batch_size].to(
                device, non_blocking=True
            )
            with autocast(enabled=True):
                feature_map = model.forward_features(micro)
                base_logits = model.forward_head(feature_map)
                attention = getattr(
                    model.ml_decoder.decoder.layers[-1], "last_attn_weights", None
                )
                if attention is None:
                    raise RuntimeError("locked model did not expose class attention")
                result = ced(feature_map, attention, base_logits, targets=None)
                private_attention = result["private_attention"].float()
                batch_count, tail_count, spatial_count = private_attention.shape
                height, width = feature_map.shape[-2:]
                if spatial_count != height * width:
                    raise RuntimeError("private attention and feature map disagree")
                soft_mask = private_attention / private_attention.amax(
                    dim=2, keepdim=True
                ).clamp_min(1e-6)
                soft_mask = soft_mask.sqrt().reshape(
                    batch_count, tail_count, 1, height, width
                ).to(feature_map)
                target_classes = ced.tail_indices.to(feature_map.device)
                verification_parts = []
                for tail_start in range(0, tail_count, 2):
                    tail_end = min(tail_start + 2, tail_count)
                    masks = soft_mask[:, tail_start:tail_end]
                    count = tail_end - tail_start
                    expanded = feature_map[:, None].expand(
                        -1, count, -1, -1, -1
                    )
                    object_scale = (
                        float(spatial_count)
                        / masks.sum(dim=(3, 4), keepdim=True).clamp_min(1.0)
                    ).clamp_max(4.0)
                    context_mask = 1.0 - masks
                    context_scale = (
                        float(spatial_count)
                        / context_mask.sum(
                            dim=(3, 4), keepdim=True
                        ).clamp_min(1.0)
                    ).clamp_max(2.0)
                    object_features = (
                        expanded * masks * object_scale
                    ).reshape(-1, *feature_map.shape[1:])
                    context_features = (
                        expanded * context_mask * context_scale
                    ).reshape(-1, *feature_map.shape[1:])
                    view_logits = model.forward_head(
                        torch.cat([object_features, context_features], dim=0)
                    )
                    object_logits, context_logits = view_logits.chunk(2, dim=0)
                    classes = target_classes[tail_start:tail_end]
                    gather_index = classes[None].expand(batch_count, -1).reshape(-1)
                    rows = torch.arange(
                        batch_count * count, device=feature_map.device
                    )
                    object_target = object_logits[rows, gather_index].reshape(
                        batch_count, count
                    )
                    context_target = context_logits[rows, gather_index].reshape(
                        batch_count, count
                    )
                    verification_parts.append(object_target - context_target)
                verification_score = torch.cat(verification_parts, dim=1)
                if not use_verification_score:
                    verification_score = torch.zeros_like(verification_score)
                single_logits = result["logits"]
                separation_score = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        result["private_attention"].float(),
                        result["shared_attention"].float(),
                        dim=2,
                        eps=1e-6,
                    )
                ).clamp(0.0, 1.0)
            current_single_logits.append(single_logits.float().cpu())
            current_logits.append(result["logits"].float().cpu())
            current_private.append(result["private_feature"].float().cpu())
            current_shared.append(result["shared_feature"].float().cpu())
            current_separation.append(separation_score.cpu())
            current_verification.append(verification_score.float().cpu())
        logits_batches.append(torch.cat(current_logits))
        single_logits_batches.append(torch.cat(current_single_logits))
        private_batches.append(torch.cat(current_private))
        shared_batches.append(torch.cat(current_shared))
        separation_batches.append(torch.cat(current_separation))
        verification_batches.append(torch.cat(current_verification))
        target_batches.append(targets.float().cpu())
    return {
        "logits": torch.cat(logits_batches),
        "single_logits": torch.cat(single_logits_batches),
        "private_features": torch.cat(private_batches).half(),
        "shared_features": torch.cat(shared_batches).half(),
        "attention_separation": torch.cat(separation_batches).half(),
        "verification_scores": torch.cat(verification_batches).half(),
        "targets": torch.cat(target_batches),
    }


def build_train_cache(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    data_config_path: Path,
    device: torch.device,
) -> tuple[Dict[str, Tensor], Mapping[str, Sequence[int]], Tensor, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config = load_data_config(data_config_path, checkpoint["config"]["dataset_key"])
    image_size = int(config["input"]["image_size"])
    train_dataset = build_lt_dataset_from_config(
        data_config, "train", image_size, verify_images=True
    )
    # Cache extraction must be deterministic; augmentation remains part of M1/M2 training only.
    train_dataset.transform = build_image_transform(image_size, training=False)
    class_split = load_class_split(data_config["class_split"], data_config["num_classes"])
    model, ced, tail_indices = build_locked_modules(
        checkpoint, train_dataset, class_split, device
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["input"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["input"]["workers"]),
        pin_memory=True,
    )
    cache = extract_cache(
        model,
        ced,
        loader,
        device,
        int(config["input"]["micro_batch_size"]),
        "Extract M2 train features",
        bool(config["input"].get("use_verification_score", True)),
    )
    cache["confounder_indices"] = ced.confounder_indices.detach().cpu().clone()
    del model, ced
    return cache, class_split, tail_indices, data_config


def metrics(targets: Tensor, logits: Tensor, class_split) -> Dict[str, Any]:
    return evaluate_multilabel_map(targets, logits, class_split)


@torch.inference_mode()
def evaluate_m3(module: nn.Module, cache: Mapping[str, Tensor], indices: Tensor, device):
    logits = []
    batch_size = 256
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
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
        )
        logits.append(result["logits"].cpu())
    return torch.cat(logits)
