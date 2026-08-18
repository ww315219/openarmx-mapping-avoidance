#!/usr/bin/env python3
"""Evaluate a residual TorchScript ensemble on unseen E8 rosbag sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.signal import savgol_filter
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--e8-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def interpolate(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    output = np.empty((target_t.size, source_y.shape[1]), dtype=np.float64)
    for column in range(source_y.shape[1]):
        output[:, column] = np.interp(target_t, source_t, source_y[:, column])
    return output


def read_bag(path: Path, joint_names: list[str]) -> tuple[dict[str, float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    required = {
        "/joint_states",
        "/openarmx/antisway/modal_state",
        "/openarmx/antisway/observer_diagnostics",
        "/experiment/antisway_phase",
    }
    missing = required - types.keys()
    if missing:
        raise RuntimeError(f"{path}: missing topics {sorted(missing)}")
    times = {topic: [] for topic in required if topic != "/experiment/antisway_phase"}
    values = {topic: [] for topic in times}
    phases: dict[str, float] = {}
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in required:
            continue
        stamp = stamp_ns * 1.0e-9
        message = deserialize_message(payload, types[topic])
        if topic == "/experiment/antisway_phase":
            phases[message.data] = stamp
        elif topic == "/joint_states":
            index = {name: position for name, position in zip(message.name, message.position)}
            if all(name in index for name in joint_names):
                times[topic].append(stamp)
                values[topic].append([index[name] for name in joint_names])
        elif topic == "/openarmx/antisway/modal_state" and len(message.data) >= 6:
            times[topic].append(stamp)
            values[topic].append(message.data[:6])
        elif topic == "/openarmx/antisway/observer_diagnostics" and len(message.data) >= 6:
            times[topic].append(stamp)
            values[topic].append(message.data[:6])
    expected_phases = {"roll_excitation", "settle_between", "yaw_excitation", "settle_after"}
    if not expected_phases.issubset(phases):
        raise RuntimeError(f"{path}: incomplete phase markers")
    arrays = {
        topic: (np.asarray(times[topic]), np.asarray(values[topic], dtype=np.float64))
        for topic in times
    }
    return phases, arrays


def session_windows(
    bag: Path,
    metadata: dict,
    history_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rate_hz = float(metadata["dataset"]["rate_hz"])
    joint_names = list(metadata["dataset"]["joint_names"])
    phases, raw = read_bag(bag, joint_names)
    start = max(series[0][0] for series in raw.values())
    end = min(series[0][-1] for series in raw.values())
    timeline = np.arange(start, end, 1.0 / rate_hz)
    position = interpolate(*raw["/joint_states"], timeline)
    state = interpolate(*raw["/openarmx/antisway/modal_state"], timeline)
    acceleration = interpolate(*raw["/openarmx/antisway/observer_diagnostics"], timeline)
    window = min(11, timeline.size if timeline.size % 2 else timeline.size - 1)
    velocity = savgol_filter(
        position,
        window_length=window,
        polyorder=min(3, window - 2),
        deriv=1,
        delta=1.0 / rate_hz,
        axis=0,
        mode="interp",
    )
    discrete_a = np.asarray(metadata["dataset"]["nominal_model"]["discrete_a"])
    discrete_b = np.asarray(metadata["dataset"]["nominal_model"]["discrete_b"])
    features = np.hstack((state, position, velocity, acceleration))[:-1]
    nominal_next = state[:-1] @ discrete_a.T + acceleration[:-1] @ discrete_b.T
    targets = state[1:] - nominal_next
    target_times = timeline[1:]
    excitation = (
        ((target_times >= phases["roll_excitation"]) & (target_times < phases["settle_between"]))
        | ((target_times >= phases["yaw_excitation"]) & (target_times < phases["settle_after"]))
    )
    valid_ends = np.flatnonzero(excitation & np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1))
    valid_ends = valid_ends[valid_ends >= history_samples - 1]
    sequences = np.stack(
        [features[end_index - history_samples + 1 : end_index + 1] for end_index in valid_ends]
    ).astype(np.float32)
    return sequences, targets[valid_ends].astype(np.float32)


def rmse(error: np.ndarray) -> tuple[float, list[float]]:
    return (
        float(np.sqrt(np.mean(np.square(error)))),
        np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    e8_dir = args.e8_dir.expanduser().resolve()
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    manifest = json.loads((model_dir / "torchscript_ensemble.json").read_text())
    history_samples = int(metadata["history_samples"])
    device = torch.device(args.device)
    models = [
        torch.jit.load(str(model_dir / name), map_location=device).eval()
        for name in manifest["models"]
    ]
    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())

    sessions = []
    all_targets = []
    all_predictions = []
    for bag in sorted(path for path in e8_dir.iterdir() if path.is_dir() and (path / "metadata.yaml").exists()):
        sequences, targets = session_windows(bag, metadata, history_samples)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(sequences), args.batch_size):
                batch = torch.from_numpy(sequences[start : start + args.batch_size]).to(device)
                ensemble = torch.stack([model(batch) for model in models])
                predictions.append(ensemble.mean(dim=0).cpu().numpy())
        prediction = np.concatenate(predictions)
        nominal_overall, nominal_per = rmse(targets)
        hybrid_overall, hybrid_per = rmse(targets - prediction)
        sessions.append(
            {
                "bag": bag.name,
                "windows": len(targets),
                "nominal_rmse": nominal_overall,
                "hybrid_rmse": hybrid_overall,
                "improvement_percent": 100.0 * (1.0 - hybrid_overall / nominal_overall),
            }
        )
        all_targets.append(targets)
        all_predictions.append(prediction)

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    nominal_overall, nominal_per = rmse(targets)
    hybrid_overall, hybrid_per = rmse(targets - predictions)
    target_names = metadata["dataset"]["target_names"]
    report = {
        "model_dir": str(model_dir),
        "e8_dir": str(e8_dir),
        "architecture": metadata["model_type"],
        "history_seconds": metadata["history_seconds"],
        "history_samples": history_samples,
        "parameter_count_per_member": parameter_count,
        "ensemble_size": len(models),
        "windows": len(targets),
        "nominal_physics_rmse": {
            "overall": nominal_overall,
            "per_state": dict(zip(target_names, nominal_per, strict=True)),
        },
        "hybrid_rmse": {
            "overall": hybrid_overall,
            "per_state": dict(zip(target_names, hybrid_per, strict=True)),
        },
        "improvement_over_nominal_percent": 100.0 * (1.0 - hybrid_overall / nominal_overall),
        "sessions": sessions,
    }
    output = args.output or model_dir / "e8_holdout_metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
