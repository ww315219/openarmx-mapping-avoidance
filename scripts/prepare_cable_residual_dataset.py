#!/usr/bin/env python3
"""Convert cable-dynamics ROS bags into a supervised residual dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.linalg import expm
from scipy.signal import savgol_filter


JOINT_NAMES = (
    "openarmx_left_joint1",
    "openarmx_left_joint2",
    "openarmx_left_joint4",
    "openarmx_right_joint1",
    "openarmx_right_joint2",
    "openarmx_right_joint4",
)
TOPICS = (
    "/joint_states",
    "/openarmx/antisway/modal_state",
    "/openarmx/antisway/observer_diagnostics",
)
FREQUENCIES_HZ = (1.6846, 2.4155)
DAMPING_RATIOS = (0.07918, 0.05082)
COUPLING = np.asarray(
    [
        [0.23935252, 0.00220909, -0.09528608, -0.23503855, -0.00061910, -0.09961852],
        [0.00217554, 0.18233809, -0.00085226, 0.01564604, 0.17299390, 0.00756648],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiment_data/cable_dynamics_residual"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiment_data/cable_dynamics_residual/learning/cable_residual_dataset.npz"
        ),
    )
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--minimum-duration", type=float, default=80.0)
    parser.add_argument(
        "--scenario-regex",
        default=r"e[456]_.*_r[123]_clean$",
        help="Only completed scenarios matching this expression are included.",
    )
    parser.add_argument(
        "--include-e7-baseline",
        action="store_true",
        help=(
            "Add completed E7 baseline bags as training-only strong-excitation "
            "sessions. Those sessions must not subsequently be reported as holdout data."
        ),
    )
    return parser.parse_args()


def discretize_model(rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    continuous_a = np.zeros((6, 6), dtype=np.float64)
    continuous_b = np.zeros((6, 6), dtype=np.float64)
    for mode_index, (frequency_hz, damping_ratio) in enumerate(
        zip(FREQUENCIES_HZ, DAMPING_RATIOS, strict=True)
    ):
        angle_index = mode_index * 2
        rate_index = angle_index + 1
        omega = 2.0 * np.pi * frequency_hz
        continuous_a[angle_index, rate_index] = 1.0
        continuous_a[rate_index, angle_index] = -(omega**2)
        continuous_a[rate_index, rate_index] = -2.0 * damping_ratio * omega
        continuous_b[rate_index, :] = COUPLING[mode_index, :]

    cutoff_omega = 2.0 * np.pi * 1.0
    continuous_a[4, 4] = -cutoff_omega
    continuous_a[5, 5] = -cutoff_omega

    augmented = np.zeros((12, 12), dtype=np.float64)
    augmented[:6, :6] = continuous_a
    augmented[:6, 6:] = continuous_b
    discrete = expm(augmented / rate_hz)
    return discrete[:6, :6], discrete[:6, 6:]


def metadata_for_session(metadata_path: Path) -> dict | None:
    session = metadata_path.parent
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    bag_path = session / str(metadata.get("bag_path", "bag"))
    if not (bag_path / "metadata.yaml").exists():
        return None
    metadata["path"] = str(session.resolve())
    metadata["bag_path_resolved"] = str(bag_path.resolve())
    return metadata


def select_sessions(args: argparse.Namespace) -> list[dict]:
    expression = re.compile(args.scenario_regex)
    candidates: list[dict] = []
    for metadata_path in sorted(args.data_root.rglob("session.json")):
        metadata = metadata_for_session(metadata_path)
        if metadata is None:
            continue
        scenario = str(metadata.get("scenario", ""))
        duration = float(
            metadata.get(
                "duration_actual_s",
                metadata.get("recorded_duration_s", metadata.get("duration", 0.0)),
            )
        )
        complete = bool(metadata.get("recording_complete", True))
        if expression.search(scenario) and complete and duration >= args.minimum_duration:
            metadata["duration_s"] = duration
            candidates.append(metadata)

    # A scenario may have been repeated after a naming mistake. Keep the earliest
    # complete recording so train/validation membership remains deterministic.
    selected: dict[str, dict] = {}
    for metadata in sorted(candidates, key=lambda item: item["path"]):
        selected.setdefault(str(metadata["scenario"]), metadata)
    sessions = list(selected.values())
    if args.include_e7_baseline:
        for metadata_path in sorted(
            args.data_root.rglob("e7_closed_loop_validation/*_baseline_metadata.json")
        ):
            bag_name = metadata_path.name.removesuffix("_metadata.json")
            bag_path = metadata_path.parent / bag_name
            if not (bag_path / "metadata.yaml").exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            sessions.append(
                {
                    **metadata,
                    "scenario": (
                        f"e7_strong_baseline_{metadata_path.parents[1].name}_{bag_name}"
                    ),
                    "path": str(metadata_path.parent.resolve()),
                    "bag_path_resolved": str(bag_path.resolve()),
                    "split_override": "train",
                }
            )
    if not sessions:
        raise RuntimeError(f"No usable sessions found below {args.data_root}")
    return sessions


def read_bag(bag_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = sorted(set(TOPICS) - topic_types.keys())
    if missing:
        raise RuntimeError(f"{bag_path}: missing topics {missing}")
    message_types = {topic: get_message(topic_types[topic]) for topic in TOPICS}
    times: dict[str, list[float]] = {topic: [] for topic in TOPICS}
    values: dict[str, list[np.ndarray]] = {topic: [] for topic in TOPICS}

    while reader.has_next():
        topic, payload, timestamp_ns = reader.read_next()
        if topic not in message_types:
            continue
        message = deserialize_message(payload, message_types[topic])
        if topic == "/joint_states":
            index = {name: i for i, name in enumerate(message.name)}
            if any(name not in index for name in JOINT_NAMES):
                continue
            value = np.asarray([message.position[index[name]] for name in JOINT_NAMES])
        else:
            value = np.asarray(message.data, dtype=np.float64)
            required = 6 if topic.endswith("modal_state") else 13
            if value.size < required:
                continue
            value = value[:required]
        times[topic].append(timestamp_ns * 1e-9)
        values[topic].append(value)

    result = {}
    for topic in TOPICS:
        if len(times[topic]) < 3:
            raise RuntimeError(f"{bag_path}: insufficient messages on {topic}")
        result[topic] = (np.asarray(times[topic]), np.vstack(values[topic]))
    return result


def interpolate(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    output = np.empty((target_t.size, source_y.shape[1]), dtype=np.float64)
    for column in range(source_y.shape[1]):
        output[:, column] = np.interp(target_t, source_t, source_y[:, column])
    return output


def resample_session(
    raw: dict[str, tuple[np.ndarray, np.ndarray]], rate_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = max(raw[topic][0][0] for topic in TOPICS)
    end = min(raw[topic][0][-1] for topic in TOPICS)
    if end - start < 5.0:
        raise RuntimeError("Overlapping topic interval is shorter than 5 seconds")
    timeline = np.arange(start, end, 1.0 / rate_hz)
    joint_position = interpolate(*raw["/joint_states"], timeline)
    modal_state = interpolate(*raw["/openarmx/antisway/modal_state"], timeline)[:, :6]
    diagnostics = interpolate(*raw["/openarmx/antisway/observer_diagnostics"], timeline)
    joint_acceleration = diagnostics[:, :6]

    window = min(11, timeline.size if timeline.size % 2 else timeline.size - 1)
    if window < 5:
        raise RuntimeError("Session is too short for velocity estimation")
    joint_velocity = savgol_filter(
        joint_position,
        window_length=window,
        polyorder=min(3, window - 2),
        deriv=1,
        delta=1.0 / rate_hz,
        axis=0,
        mode="interp",
    )
    return timeline, modal_state, joint_position, joint_velocity, joint_acceleration


def main() -> None:
    args = parse_args()
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive")
    sessions = select_sessions(args)
    discrete_a, discrete_b = discretize_model(args.rate_hz)

    feature_blocks = []
    target_blocks = []
    nominal_blocks = []
    actual_blocks = []
    time_blocks = []
    session_blocks = []
    split_blocks = []
    exported_sessions = []

    for session_index, metadata in enumerate(sessions):
        session = Path(metadata["path"])
        raw = read_bag(Path(metadata["bag_path_resolved"]))
        timeline, state, position, velocity, acceleration = resample_session(raw, args.rate_hz)
        feature = np.hstack((state, position, velocity, acceleration))[:-1]
        nominal_next = state[:-1] @ discrete_a.T + acceleration[:-1] @ discrete_b.T
        actual_next = state[1:]
        target = actual_next - nominal_next
        valid = np.isfinite(feature).all(axis=1) & np.isfinite(target).all(axis=1)
        validation = (
            metadata.get("split_override") == "validation"
            or (
                metadata.get("split_override") is None
                and bool(re.search(r"_r3_clean$", str(metadata["scenario"])))
            )
        )

        feature_blocks.append(feature[valid].astype(np.float32))
        target_blocks.append(target[valid].astype(np.float32))
        nominal_blocks.append(nominal_next[valid].astype(np.float32))
        actual_blocks.append(actual_next[valid].astype(np.float32))
        time_blocks.append((timeline[:-1][valid] - timeline[0]).astype(np.float32))
        session_blocks.append(np.full(valid.sum(), session_index, dtype=np.int32))
        split_blocks.append(np.full(valid.sum(), int(validation), dtype=np.uint8))
        exported_sessions.append(
            {
                "index": session_index,
                "scenario": metadata["scenario"],
                "path": metadata["path"],
                "split": "validation" if validation else "train",
                "samples": int(valid.sum()),
            }
        )
        print(
            f"[{exported_sessions[-1]['split']}] {metadata['scenario']}: "
            f"{valid.sum()} samples"
        )

    feature_names = (
        [f"state_{name}" for name in ("roll_angle", "roll_rate", "yaw_angle", "yaw_rate", "slow_gyro_x", "slow_gyro_z")]
        + [f"position_{name}" for name in JOINT_NAMES]
        + [f"velocity_{name}" for name in JOINT_NAMES]
        + [f"acceleration_{name}" for name in JOINT_NAMES]
    )
    target_names = [
        "roll_angle_residual",
        "roll_rate_residual",
        "yaw_angle_residual",
        "yaw_rate_residual",
        "slow_gyro_x_residual",
        "slow_gyro_z_residual",
    ]
    dataset_metadata = {
        "format_version": 1,
        "rate_hz": args.rate_hz,
        "feature_names": feature_names,
        "target_names": target_names,
        "joint_names": JOINT_NAMES,
        "nominal_model": {
            "frequencies_hz": FREQUENCIES_HZ,
            "damping_ratios": DAMPING_RATIOS,
            "coupling": COUPLING.tolist(),
            "discrete_a": discrete_a.tolist(),
            "discrete_b": discrete_b.tolist(),
        },
        "sessions": exported_sessions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(feature_blocks),
        targets=np.concatenate(target_blocks),
        nominal_next=np.concatenate(nominal_blocks),
        actual_next=np.concatenate(actual_blocks),
        time_s=np.concatenate(time_blocks),
        session_index=np.concatenate(session_blocks),
        split=np.concatenate(split_blocks),
        metadata_json=np.asarray(json.dumps(dataset_metadata)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    train_count = int(sum(block.size - block.sum() for block in split_blocks))
    validation_count = int(sum(block.sum() for block in split_blocks))
    print(f"Wrote {args.output}")
    print(f"train={train_count} validation={validation_count} features={len(feature_names)}")


if __name__ == "__main__":
    main()
