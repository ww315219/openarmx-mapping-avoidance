#!/usr/bin/env python3
"""Train and evaluate the GRU/Attention-GRU history-length ablation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiment_data/cable_dynamics_residual/learning/cable_residual_dataset_attention.npz"),
    )
    parser.add_argument("--e8-dir", type=Path)
    parser.add_argument("--history-seconds", default="0.5,1.0,1.5")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment_data/cable_dynamics_residual/learning/ablations"),
    )
    return parser.parse_args()


def newest_e8(root: Path) -> Path:
    candidates = sorted(root.glob("e8_holdout_pairs_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No E8 holdout directory below {root}")
    return candidates[0]


def run(command: list[str], log_path: Path) -> None:
    print("[ablation]", " ".join(command), flush=True)
    environment = os.environ.copy()
    ros_library = "/opt/ros/jazzy/lib"
    environment["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in (ros_library, environment.get("LD_LIBRARY_PATH", ""))
        if part
    )
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def main() -> None:
    args = parse_args()
    histories = [float(value) for value in args.history_seconds.split(",")]
    if any(value <= 0.0 or value > 3.0 for value in histories):
        raise ValueError("history lengths must be in (0, 3] seconds")
    workspace = Path(__file__).resolve().parents[1]
    dataset = args.dataset.expanduser().resolve()
    e8_dir = (
        args.e8_dir.expanduser().resolve()
        if args.e8_dir is not None
        else newest_e8(workspace / "experiment_data/cable_dynamics_residual")
    )
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = args.output_root.expanduser().resolve() / f"attention_gru_ablation_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "dataset": str(dataset),
        "e8_dir": str(e8_dir),
        "histories_seconds": histories,
        "architectures": ["gru", "attention_gru"],
        "ensemble_size": args.ensemble_size,
        "seed_start": 42,
        "runs": [],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    trainer = workspace / "scripts/train_cable_residual_gru.py"
    evaluator = workspace / "scripts/evaluate_residual_model_on_e8.py"

    try:
        for history in histories:
            for architecture in ("gru", "attention_gru"):
                label = f"{architecture}_h{history:.1f}".replace(".", "p")
                model_dir = output / label
                train_command = [
                    sys.executable,
                    str(trainer),
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(model_dir),
                    "--history-seconds",
                    str(history),
                    "--architecture",
                    architecture,
                    "--hidden-size",
                    "48",
                    "--attention-size",
                    "24",
                    "--attention-strength",
                    "0.35",
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--ensemble-size",
                    str(args.ensemble_size),
                    "--seed",
                    "42",
                    "--device",
                    args.device,
                ]
                run(train_command, output / f"{label}_train.log")
                evaluation_command = [
                    sys.executable,
                    str(evaluator),
                    "--model-dir",
                    str(model_dir),
                    "--e8-dir",
                    str(e8_dir),
                    "--device",
                    args.device,
                ]
                run(evaluation_command, output / f"{label}_e8.log")
                manifest["runs"].append(
                    {
                        "label": label,
                        "architecture": architecture,
                        "history_seconds": history,
                        "model_dir": str(model_dir),
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        rows = []
        for item in manifest["runs"]:
            model_dir = Path(item["model_dir"])
            validation = json.loads((model_dir / "metrics.json").read_text())
            e8 = json.loads((model_dir / "e8_holdout_metrics.json").read_text())
            rows.append(
                {
                    "architecture": item["architecture"],
                    "history_seconds": item["history_seconds"],
                    "validation_rmse": validation["gru_residual_rmse"]["overall"],
                    "validation_improvement_percent": validation["gru_improvement_over_nominal_percent"],
                    "e8_rmse": e8["hybrid_rmse"]["overall"],
                    "e8_improvement_percent": e8["improvement_over_nominal_percent"],
                    "parameter_count": e8["parameter_count_per_member"],
                }
            )
        comparisons = []
        for history in histories:
            gru = next(row for row in rows if row["architecture"] == "gru" and row["history_seconds"] == history)
            attention = next(row for row in rows if row["architecture"] == "attention_gru" and row["history_seconds"] == history)
            comparisons.append(
                {
                    "history_seconds": history,
                    "attention_vs_gru_validation_improvement_percent": 100.0 * (1.0 - attention["validation_rmse"] / gru["validation_rmse"]),
                    "attention_vs_gru_e8_improvement_percent": 100.0 * (1.0 - attention["e8_rmse"] / gru["e8_rmse"]),
                }
            )
        summary = {"rows": rows, "paired_comparisons": comparisons}
        (output / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output / "ablation_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "complete"
        manifest["summary"] = str(output / "ablation_summary.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"[ablation] complete: {output}")
    except Exception as exc:
        manifest["status"] = "aborted"
        manifest["error"] = str(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
