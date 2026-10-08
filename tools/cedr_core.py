#!/usr/bin/env python3
"""Construction and compatibility helpers for the public two-module CEDR model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from src.methods.cedr import CEDR, EvidenceDecoupling, MemoryReliabilityRefinement
from src.models import build_cider_base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_visual_model(
    model_config: Mapping[str, Any], state: Mapping[str, Tensor], device: torch.device
) -> nn.Module:
    model = build_cider_base(
        num_classes=int(model_config["num_classes"]),
        pretrained=False,
        num_groups=int(model_config["num_groups"]),
        decoder_embedding=int(model_config["decoder_embedding"]),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_evidence_decoupling(
    num_classes: int,
    feature_dim: int,
    config: Mapping[str, Any],
    state: Mapping[str, Tensor],
) -> EvidenceDecoupling:
    module = EvidenceDecoupling(
        num_classes=num_classes,
        tail_indices=state["tail_indices"].long(),
        confounder_indices=state["confounder_indices"].long(),
        confounder_weights=state["confounder_weights"].float(),
        tail_positive_weights=state["tail_positive_weights"].float(),
        feature_dim=feature_dim,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        confounder_strength=float(config["confounder_strength"]),
        private_residual_floor=float(config["private_residual_floor"]),
        maximum_logit_correction=float(config["maximum_logit_correction"]),
        initial_gate_logit=float(config["initial_gate_logit"]),
        negative_loss_weight=float(config["negative_loss_weight"]),
        ranking_margin=float(config["ranking_margin"]),
        evidence_margin=float(config["evidence_margin"]),
        hard_negative_count=int(config["hard_negative_count"]),
        tail_bce_weight=float(config["tail_bce_weight"]),
        ranking_weight=float(config["ranking_weight"]),
        evidence_margin_weight=float(config["evidence_margin_weight"]),
        context_suppression_weight=float(config["context_suppression_weight"]),
        attention_overlap_weight=float(config["attention_overlap_weight"]),
        feature_orthogonality_weight=float(config["feature_orthogonality_weight"]),
        correction_regularization_weight=float(config["correction_regularization_weight"]),
        routing_mode=str(config.get("routing_mode", "legacy")),
        shared_evidence_strength=float(config.get("shared_evidence_strength", 0.5)),
        route_prior_scale=float(config.get("route_prior_scale", 0.5)),
        route_overlap_scale=float(config.get("route_overlap_scale", 4.0)),
        route_confidence_scale=float(config.get("route_confidence_scale", 1.0)),
        route_temperature=float(config.get("route_temperature", 0.5)),
        route_overlap_center=float(config.get("route_overlap_center", 0.5)),
        null_route_initial_logit=float(config.get("null_route_initial_logit", 1.0)),
        route_supervision_weight=float(config.get("route_supervision_weight", 0.0)),
    )
    module.load_state_dict(state, strict=True)
    return module


def build_memory_refinement(
    num_classes: int,
    config: Mapping[str, Any],
    state: Mapping[str, Tensor],
    confounder_indices: Tensor,
) -> MemoryReliabilityRefinement:
    method = config
    module = MemoryReliabilityRefinement(
        num_classes=num_classes,
        tail_indices=state["tail_indices"].long(),
        positive_memory=state["positive_memory"].float(),
        negative_memory=state["negative_memory"].float(),
        positive_valid=state["positive_valid"].bool(),
        negative_valid=state["negative_valid"].bool(),
        maximum_residual=float(method["maximum_residual"]),
        uncertainty_temperature=float(method["uncertainty_temperature"]),
        ranking_margin=float(method["ranking_margin"]),
        ranking_weight=float(method["ranking_weight"]),
        residual_regularization_weight=float(method["residual_regularization_weight"]),
        adapter_deviation=float(method.get("adapter_deviation", 0.25)),
        adapter_regularization_weight=float(method.get("adapter_regularization_weight", 0.0005)),
        adapter_initial_temperature=float(method.get("adapter_initial_temperature", 5.0)),
        adapter_classification_weight=float(method.get("adapter_classification_weight", 1.0)),
        adapter_ranking_weight=float(method.get("adapter_ranking_weight", 0.25)),
        label_graph=state["label_graph"].float(),
        class_prior=state["class_prior"].float(),
        dependency_deviation=float(method.get("dependency_deviation", 0.5)),
        dependency_initial_temperature=float(method.get("dependency_initial_temperature", 3.0)),
        dependency_initial_scale=float(method.get("dependency_initial_scale", 0.0)),
        dependency_classification_weight=float(method.get("dependency_classification_weight", 1.0)),
        dependency_ranking_weight=float(method.get("dependency_ranking_weight", 0.25)),
        dependency_regularization_weight=float(method.get("dependency_regularization_weight", 0.0005)),
        dependency_private_gate_temperature=float(method.get("dependency_private_gate_temperature", 3.0)),
        shared_positive_memory=state["shared_positive_memory"].float(),
        shared_negative_memory=state["shared_negative_memory"].float(),
        shared_positive_valid=state["shared_positive_valid"].bool(),
        shared_negative_valid=state["shared_negative_valid"].bool(),
        shared_excess_strength=float(method.get("shared_excess_strength", 1.0)),
        causal_initial_temperature=float(method.get("causal_initial_temperature", 5.0)),
        causal_classification_weight=float(method.get("causal_classification_weight", 1.0)),
        causal_ranking_weight=float(method.get("causal_ranking_weight", 0.5)),
        attention_gate_strength=float(method.get("attention_gate_strength", 1.0)),
        verification_mean=state["verification_mean"].float(),
        verification_std=state["verification_std"].float(),
        verification_initial_temperature=float(method.get("verification_initial_temperature", 3.0)),
        verification_classification_weight=float(method.get("verification_classification_weight", 1.0)),
        verification_ranking_weight=float(method.get("verification_ranking_weight", 0.5)),
        positive_gate_mode=str(method.get("positive_gate_mode", "legacy")),
        prototype_margin=float(method.get("prototype_margin", 0.0)),
        private_shared_gap_margin=float(method.get("private_shared_gap_margin", 0.0)),
        private_confidence_threshold=float(method.get("private_confidence_threshold", 0.2)),
        positive_gate_temperature=float(method.get("positive_gate_temperature", 4.0)),
        uncertainty_dead_zone=float(method.get("uncertainty_dead_zone", 0.0)),
        uncertainty_gate_temperature=float(method.get("uncertainty_gate_temperature", 4.0)),
        maximum_positive_residual=float(method.get("maximum_positive_residual", method["maximum_residual"])),
        maximum_negative_residual=float(method.get("maximum_negative_residual", method["maximum_residual"])),
        confounder_indices=confounder_indices.long(),
        common_head_ranking_weight=float(method.get("common_head_ranking_weight", 0.0)),
        common_head_ranking_margin=float(method.get("common_head_ranking_margin", 1.0)),
        dependency_enabled=bool(method.get("dependency_enabled", True)),
    )
    module.load_state_dict(state, strict=True)
    return module


def load_cedr_composite(
    checkpoint_or_path: Mapping[str, Any] | Path,
    device: torch.device,
) -> tuple[nn.Module, CEDR]:
    checkpoint = (
        torch.load(checkpoint_or_path, map_location="cpu")
        if isinstance(checkpoint_or_path, Path)
        else checkpoint_or_path
    )
    if checkpoint.get("format_version") != "two_module_v1":
        raise ValueError("unsupported two-module checkpoint format")
    if checkpoint.get("module_name") != "CEDR":
        raise ValueError("checkpoint is not a CEDR composite")
    config = checkpoint["config"]
    model_config = config["model"]
    model = build_visual_model(model_config, checkpoint["lci_or_visual_state"], device)
    decoupling = build_evidence_decoupling(
        int(model_config["num_classes"]),
        int(model_config["feature_dim"]),
        config["decoupling"],
        checkpoint["decoupling_state"],
    )
    refinement = build_memory_refinement(
        int(model_config["num_classes"]),
        config["memory_refinement"]["method"],
        checkpoint["memory_refinement_state"],
        checkpoint["decoupling_state"]["confounder_indices"],
    )
    cedr = CEDR(decoupling, refinement, checkpoint["selected_residual_alpha"])
    return model, cedr.to(device).eval()


@torch.inference_mode()
def score_images(model: nn.Module, cedr: CEDR, images: Tensor) -> Mapping[str, Any]:
    feature = model.forward_features(images)
    baseline_logits = model.forward_head(feature).float()
    attention = model.ml_decoder.decoder.layers[-1].last_attn_weights
    return cedr(feature, attention, baseline_logits)
