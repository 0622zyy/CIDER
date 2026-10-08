#!/usr/bin/env python3
"""Wrap locked legacy checkpoints in the public two-module format without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cedr_core import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--decoupling", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-config", type=Path, default=None)
    parser.add_argument("--training-output-root", type=Path, default=None)
    args = parser.parse_args()

    m1_path = args.m1.resolve()
    dec_path = args.decoupling.resolve()
    ref_path = args.refinement.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    m1 = torch.load(m1_path, map_location="cpu")
    dec = torch.load(dec_path, map_location="cpu")
    ref = torch.load(ref_path, map_location="cpu")
    hashes = {
        "m1": sha256_file(m1_path),
        "decoupling": sha256_file(dec_path),
        "refinement": sha256_file(ref_path),
    }
    recorded = ref.get("checkpoint_sha256")
    if recorded is not None and recorded != hashes["decoupling"]:
        raise RuntimeError("refinement checkpoint does not belong to the supplied decoupling checkpoint")
    tail_indices = dec["ced"]["tail_indices"].long()
    if not torch.equal(tail_indices, ref["module3"]["tail_indices"].long()):
        raise RuntimeError("Tail index mapping changed between internal phases")

    m1_payload = {
        "format_version": "two_module_v1",
        "module_name": "LCI",
        "lci_or_visual_state": m1["model"],
        "config": m1["config"],
        "source_checkpoints": {"m1": str(m1_path)},
        "source_sha256": {"m1": hashes["m1"]},
        "source_epoch": m1.get("epoch"),
    }
    m1_out = output / "m1_two_module.pth"
    torch.save(m1_payload, m1_out)

    ref_config = ref["config"]
    training_root = (
        args.training_output_root.resolve()
        if args.training_output_root is not None
        else output.parent / "training"
    )
    combined_config = {
        "module": "CEDR",
        "experiment_name": f"{dec['config']['dataset_key']}_cedr_seed42",
        "dataset_key": dec["config"]["dataset_key"],
        "seed": int(dec["config"]["seed"]),
        "model": dec["config"]["model"],
        "loss": dec["config"].get("loss", {}),
        "lci": dec["config"].get("lci", {}),
        "decoupling": dec["config"]["ced"],
        "memory_refinement": {
            "output_dir": str(training_root / "memory_refinement_cache"),
            "final_output_dir": str(training_root / "memory_refinement_final"),
            "checkpoint_selection": ref_config.get("checkpoint_selection", "best_official_test_tail_map"),
            "ablation_variant": ref_config.get("ablation_variant", "t1_original_gate"),
            "input": ref_config["input"],
            "memory": ref_config["memory"],
            "method": ref_config["method"],
        },
        "training": {
            "decoupling": dec["config"].get("training", {}),
            "memory_refinement": ref_config.get("training", {}),
        },
        "phase_order": ["decoupling", "memory_refinement"],
        "acceptance": dec["config"].get("acceptance", {}),
        "output_root": str(training_root / "decoupling"),
        "module1_result_path": dec["config"].get("module1_result_path"),
        "deterministic_protocol": dec["config"].get("deterministic_protocol", {}),
        "compatibility_sources": {
            "decoupling_checkpoint": str(dec_path),
            "memory_refinement_checkpoint": str(ref_path),
        },
    }
    composite = {
        "format_version": "two_module_v1",
        "module_name": "CEDR",
        "lci_or_visual_state": dec["model"],
        "decoupling_state": dec["ced"],
        "memory_refinement_state": ref["module3"],
        "tail_indices": tail_indices,
        "selected_residual_alpha": float(ref.get("selected_residual_alpha", 1.0)),
        "config": combined_config,
        "source_checkpoints": {
            "m1": str(m1_path),
            "decoupling": str(dec_path),
            "memory_refinement": str(ref_path),
        },
        "source_sha256": hashes,
        "source_selection": {
            "decoupling_epoch": dec.get("epoch"),
            "memory_refinement_epoch": ref.get("selected_epoch", ref.get("fixed_epochs")),
            "residual_alpha": float(ref.get("selected_residual_alpha", 1.0)),
        },
    }
    m2_out = output / "m2_cedr_composite.pth"
    torch.save(composite, m2_out)
    manifest = {
        "format_version": "two_module_v1",
        "m1_checkpoint": str(m1_out),
        "m1_sha256": sha256_file(m1_out),
        "m2_checkpoint": str(m2_out),
        "m2_sha256": sha256_file(m2_out),
        "source_checkpoints": composite["source_checkpoints"],
        "source_sha256": hashes,
        "training_performed": False,
    }
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.public_config is not None:
        args.public_config.parent.mkdir(parents=True, exist_ok=True)
        args.public_config.write_text(
            json.dumps(combined_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
