#!/usr/bin/env python3
"""Train an ensemble of compact GRU cable-dynamics residual models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "experiment_data/cable_dynamics_residual/learning/cable_residual_dataset.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--history-seconds", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument(
        "--architecture",
        choices=("gru", "attention_gru"),
        default="attention_gru",
    )
    parser.add_argument("--attention-size", type=int, default=24)
    parser.add_argument("--attention-strength", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        history: int,
    ) -> None:
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.indices = indices.astype(np.int64)
        self.history = history

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.indices[item]) + 1
        return self.features[end - self.history : end], self.targets[end - 1]


class ResidualGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        attention_size: int = 0,
        attention_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.use_attention = attention_size > 0 and attention_strength > 0.0
        self.attention_strength = float(attention_strength)
        effective_attention_size = max(1, attention_size)
        self.attention_projection = nn.Linear(
            hidden_size, effective_attention_size
        )
        self.attention_score = nn.Linear(
            effective_attention_size, 1, bias=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(sequence)
        representation = output[:, -1, :]
        if self.use_attention:
            logits = self.attention_score(
                torch.tanh(self.attention_projection(output))
            )
            weights = torch.softmax(logits, dim=1)
            context = torch.sum(weights * output, dim=1)
            representation = representation + self.attention_strength * (
                context - representation
            )
        return self.head(representation)


class DeployableResidualGRU(nn.Module):
    def __init__(
        self,
        model: ResidualGRU,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("feature_mean", torch.from_numpy(feature_mean))
        self.register_buffer("feature_std", torch.from_numpy(feature_std))
        self.register_buffer("target_mean", torch.from_numpy(target_mean))
        self.register_buffer("target_std", torch.from_numpy(target_std))

    def forward(self, raw_sequence: torch.Tensor) -> torch.Tensor:
        normalized = (raw_sequence - self.feature_mean) / self.feature_std
        return self.model(normalized) * self.target_std + self.target_mean


def valid_window_indices(
    session_index: np.ndarray, split: np.ndarray, split_value: int, history: int
) -> np.ndarray:
    valid = []
    for session in np.unique(session_index[split == split_value]):
        indices = np.flatnonzero((session_index == session) & (split == split_value))
        if indices.size < history:
            continue
        # Prepared sessions are contiguous, but this check protects the window contract.
        for end in indices[history - 1 :]:
            window = np.arange(end - history + 1, end + 1)
            if np.all(session_index[window] == session) and np.all(split[window] == split_value):
                valid.append(end)
    return np.asarray(valid, dtype=np.int64)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            prediction = model(features)
            total_loss += float(criterion(prediction, target).cpu())
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    return total_loss / target_array.size, prediction_array, target_array


def train_one(
    seed: int,
    args: argparse.Namespace,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    input_size: int,
    output_size: int,
    device: torch.device,
) -> tuple[ResidualGRU, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    attention_size = args.attention_size if args.architecture == "attention_gru" else 0
    attention_strength = (
        args.attention_strength if args.architecture == "attention_gru" else 0.0
    )
    model = ResidualGRU(
        input_size,
        args.hidden_size,
        output_size,
        attention_size=attention_size,
        attention_strength=attention_strength,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    best_state = None
    best_validation = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * target.shape[0]
            train_count += target.shape[0]

        validation_loss, _, _ = evaluate(model, validation_loader, device)
        train_loss /= max(train_count, 1)
        history.append(
            {"seed": seed, "epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss}
        )
        if validation_loss < best_validation - 1.0e-7:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"seed={seed} epoch={epoch:03d} train_mse={train_loss:.6f} "
                f"validation_mse={validation_loss:.6f}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"seed={seed} early_stop={epoch} best_validation_mse={best_validation:.6f}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return model, history


def rmse(error: np.ndarray) -> tuple[float, list[float]]:
    per_output = np.sqrt(np.mean(np.square(error), axis=0))
    return float(np.sqrt(np.mean(np.square(error)))), per_output.tolist()


def export_numpy_ensemble(
    output_path: Path,
    models: list[ResidualGRU],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    history_samples: int,
    rate_hz: float,
) -> None:
    state_dicts = [model.state_dict() for model in models]

    def stacked(name: str) -> np.ndarray:
        return np.stack(
            [state[name].detach().cpu().numpy() for state in state_dicts]
        ).astype(np.float32)

    np.savez_compressed(
        output_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        history_samples=np.asarray([history_samples], dtype=np.int32),
        rate_hz=np.asarray([rate_hz], dtype=np.float32),
        weight_ih=stacked("gru.weight_ih_l0"),
        weight_hh=stacked("gru.weight_hh_l0"),
        bias_ih=stacked("gru.bias_ih_l0"),
        bias_hh=stacked("gru.bias_hh_l0"),
        head0_weight=stacked("head.0.weight"),
        head0_bias=stacked("head.0.bias"),
        head2_weight=stacked("head.2.weight"),
        head2_bias=stacked("head.2.bias"),
    )


def main() -> None:
    args = parse_args()
    if args.attention_size <= 0:
        raise ValueError("--attention-size must be positive")
    if not 0.0 <= args.attention_strength <= 1.0:
        raise ValueError("--attention-strength must be in [0, 1]")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = args.dataset.parent / "models" / f"cable_residual_gru_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = choose_device(args.device)
    print(f"device={device} output={args.output_dir}")

    archive = np.load(args.dataset, allow_pickle=False)
    features = archive["features"].astype(np.float32)
    targets = archive["targets"].astype(np.float32)
    session_index = archive["session_index"]
    split = archive["split"]
    metadata = json.loads(str(archive["metadata_json"]))
    rate_hz = float(metadata["rate_hz"])
    history_samples = max(2, int(round(args.history_seconds * rate_hz)))

    train_rows = split == 0
    feature_mean = features[train_rows].mean(axis=0).astype(np.float32)
    feature_std = features[train_rows].std(axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, 1.0e-6)
    target_mean = targets[train_rows].mean(axis=0).astype(np.float32)
    target_std = targets[train_rows].std(axis=0).astype(np.float32)
    target_std = np.maximum(target_std, 1.0e-6)
    normalized_features = ((features - feature_mean) / feature_std).astype(np.float32)
    normalized_targets = ((targets - target_mean) / target_std).astype(np.float32)

    train_indices = valid_window_indices(session_index, split, 0, history_samples)
    validation_indices = valid_window_indices(session_index, split, 1, history_samples)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise RuntimeError("The dataset does not contain usable train and validation windows")
    print(
        f"history={history_samples} samples ({history_samples / rate_hz:.3f}s) "
        f"train_windows={train_indices.size} validation_windows={validation_indices.size}"
    )

    train_dataset = SequenceDataset(
        normalized_features, normalized_targets, train_indices, history_samples
    )
    validation_dataset = SequenceDataset(
        normalized_features, normalized_targets, validation_indices, history_samples
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    models = []
    torchscript_names = []
    all_history = []
    normalized_predictions = []
    normalized_validation_targets = None
    for ensemble_index in range(args.ensemble_size):
        seed = args.seed + ensemble_index
        model, training_history = train_one(
            seed,
            args,
            train_loader,
            validation_loader,
            features.shape[1],
            targets.shape[1],
            device,
        )
        _, prediction, validation_target = evaluate(model, validation_loader, device)
        normalized_predictions.append(prediction)
        normalized_validation_targets = validation_target
        model = model.cpu().eval()
        deployable = DeployableResidualGRU(
            model, feature_mean, feature_std, target_mean, target_std
        ).eval()
        scripted = torch.jit.script(deployable)
        model_name = f"residual_gru_seed_{seed}.pt"
        scripted.save(str(args.output_dir / model_name))
        torchscript_names.append(model_name)
        models.append(model)
        all_history.extend(training_history)

    ensemble_prediction = np.mean(np.stack(normalized_predictions), axis=0)
    prediction_physical = ensemble_prediction * target_std + target_mean
    validation_target_physical = normalized_validation_targets * target_std + target_mean

    # Linear current-state residual baseline. It deliberately has no temporal memory.
    train_x = np.hstack((normalized_features[train_indices], np.ones((train_indices.size, 1))))
    train_y = normalized_targets[train_indices]
    ridge = 1.0e-3 * np.eye(train_x.shape[1])
    linear_weights = np.linalg.solve(train_x.T @ train_x + ridge, train_x.T @ train_y)
    validation_x = np.hstack(
        (normalized_features[validation_indices], np.ones((validation_indices.size, 1)))
    )
    linear_prediction = (validation_x @ linear_weights) * target_std + target_mean

    nominal_overall, nominal_per_state = rmse(validation_target_physical)
    linear_overall, linear_per_state = rmse(validation_target_physical - linear_prediction)
    gru_overall, gru_per_state = rmse(validation_target_physical - prediction_physical)
    target_names = metadata["target_names"]
    metrics = {
        "validation_windows": int(validation_indices.size),
        "nominal_physics_rmse": {
            "overall": nominal_overall,
            "per_state": dict(zip(target_names, nominal_per_state, strict=True)),
        },
        "linear_residual_rmse": {
            "overall": linear_overall,
            "per_state": dict(zip(target_names, linear_per_state, strict=True)),
        },
        "gru_residual_rmse": {
            "overall": gru_overall,
            "per_state": dict(zip(target_names, gru_per_state, strict=True)),
        },
        "gru_improvement_over_nominal_percent": 100.0 * (1.0 - gru_overall / nominal_overall),
        "gru_improvement_over_linear_percent": 100.0 * (1.0 - gru_overall / linear_overall),
    }
    export_metadata = {
        "model_type": f"physics_residual_{args.architecture}_ensemble",
        "ensemble_size": args.ensemble_size,
        "history_samples": history_samples,
        "history_seconds": history_samples / rate_hz,
        "hidden_size": args.hidden_size,
        "attention_size": args.attention_size if args.architecture == "attention_gru" else 0,
        "attention_strength": (
            args.attention_strength if args.architecture == "attention_gru" else 0.0
        ),
        "input_contract": "[batch, history_samples, 24] raw physical units",
        "output_contract": "[batch, 6] one-step modal-state residual",
        "dataset": metadata,
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
        },
        "metrics": metrics,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(export_metadata, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "epoch", "train_mse", "validation_mse"))
        writer.writeheader()
        writer.writerows(all_history)
    np.save(args.output_dir / "linear_residual_weights.npy", linear_weights.astype(np.float32))
    torchscript_manifest = {
        "format_version": 1,
        "backend": "torchscript",
        "models": torchscript_names,
        "metadata": "model_metadata.json",
    }
    (args.output_dir / "torchscript_ensemble.json").write_text(
        json.dumps(torchscript_manifest, indent=2), encoding="utf-8"
    )
    if args.architecture == "gru":
        export_numpy_ensemble(
            args.output_dir / "residual_gru_ensemble_numpy.npz",
            models,
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            history_samples,
            rate_hz,
        )
    print(json.dumps(metrics, indent=2))
    print(f"Models and report written to {args.output_dir}")


if __name__ == "__main__":
    main()
