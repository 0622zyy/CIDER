#!/usr/bin/env python3
"""Public trainer for M2/CEDR with two unchanged sequential internal phases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decoupling_phase_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": config["experiment_name"],
        "dataset_key": config["dataset_key"],
        "seed": config["seed"],
        "model": config["model"],
        "loss": config["loss"],
        "training": config["training"]["decoupling"],
        "lci": config["lci"],
        "ced": config["decoupling"],
        "acceptance": config.get("acceptance", {}),
        "output_root": config["output_root"],
        "module1_result_path": config["module1_result_path"],
        "deterministic_protocol": config.get("deterministic_protocol", {}),
    }


def refinement_phase_config(config: Mapping[str, Any]) -> dict[str, Any]:
    ref = config["memory_refinement"]
    return {
        "seed": config["seed"],
        "output_dir": ref["output_dir"],
        "checkpoint_selection": ref.get("checkpoint_selection", "best_official_test_tail_map"),
        "input": ref["input"],
        "memory": ref["memory"],
        "method": ref["method"],
        "training": config["training"]["memory_refinement"],
        "ablation_variant": ref.get("ablation_variant", "t1_original_gate"),
    }


def run(command: list[str]) -> None:
    print("CEDR phase command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("decoupling", "refinement", "all"), default="all"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Locked phase-A checkpoint for phase=refinement; phase=all uses the newly generated checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-memory-refinement", type=Path, default=None)
    args = parser.parse_args()
    config = load_json(args.config)
    if config.get("module") != "CEDR":
        raise ValueError("M2 config must declare module=CEDR")
    if list(config.get("phase_order", [])) != ["decoupling", "memory_refinement"]:
        raise ValueError("CEDR phase order must be decoupling then memory_refinement")

    with tempfile.TemporaryDirectory(prefix="cider_cedr_") as temporary:
        temporary_path = Path(temporary)
        dec_path = temporary_path / "decoupling.json"
        ref_path = temporary_path / "memory_refinement.json"
        dec_path.write_text(
            json.dumps(decoupling_phase_config(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        ref_path.write_text(
            json.dumps(refinement_phase_config(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        python = sys.executable
        if args.phase in {"decoupling", "all"}:
            run(
                [
                    python,
                    "-m",
                    "tools._internal.train_decoupling",
                    "--config",
                    str(dec_path),
                    "--data-config",
                    str(args.data_config),
                ]
            )
        if args.phase in {"refinement", "all"}:
            checkpoint = args.checkpoint
            if checkpoint is None:
                checkpoint = (
                    Path(config["output_root"])
                    / str(config["experiment_name"])
                    / "lci_ced_best.pth"
                )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing locked decoupling checkpoint: {checkpoint}")
            command = [
                python,
                "-m",
                "tools._internal.train_refinement",
                "--config",
                str(ref_path),
                "--data-config",
                str(args.data_config),
                "--checkpoint",
                str(checkpoint),
                "--name",
                str(config["memory_refinement"]["final_output_dir"]),
                "--device",
                args.device,
                "--variant",
                str(config["memory_refinement"].get("ablation_variant", "t1_original_gate")),
            ]
            if args.init_memory_refinement is not None:
                command.extend(["--init-module3", str(args.init_memory_refinement)])
            run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
