"""Small NumPy runtime for exported PyTorch GRU residual ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


class NumpyResidualGRUEnsemble:
    """Evaluate a one-layer PyTorch-compatible GRU ensemble."""

    def __init__(self, archive_path: str | Path) -> None:
        archive = np.load(Path(archive_path), allow_pickle=False)
        required = {
            "feature_mean",
            "feature_std",
            "target_mean",
            "target_std",
            "weight_ih",
            "weight_hh",
            "bias_ih",
            "bias_hh",
            "head0_weight",
            "head0_bias",
            "head2_weight",
            "head2_bias",
            "history_samples",
            "rate_hz",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Residual model archive is missing {missing}")
        for name in required:
            setattr(self, name, np.asarray(archive[name], dtype=np.float64))
        self.history_samples = int(self.history_samples.reshape(-1)[0])
        self.rate_hz = float(self.rate_hz.reshape(-1)[0])
        self.ensemble_size = int(self.weight_ih.shape[0])
        self.input_size = int(self.feature_mean.size)
        self.output_size = int(self.target_mean.size)
        self.hidden_size = int(self.weight_hh.shape[-1])
        if self.weight_ih.shape != (
            self.ensemble_size,
            3 * self.hidden_size,
            self.input_size,
        ):
            raise ValueError("Invalid GRU input-weight dimensions")

    def predict(self, raw_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence = np.asarray(raw_sequence, dtype=np.float64)
        if sequence.shape != (self.history_samples, self.input_size):
            raise ValueError(
                f"Expected sequence {(self.history_samples, self.input_size)}, "
                f"received {sequence.shape}"
            )
        sequence = (sequence - self.feature_mean) / self.feature_std
        predictions = np.empty(
            (self.ensemble_size, self.output_size), dtype=np.float64
        )
        for ensemble_index in range(self.ensemble_size):
            hidden = np.zeros(self.hidden_size, dtype=np.float64)
            weight_ih = self.weight_ih[ensemble_index]
            weight_hh = self.weight_hh[ensemble_index]
            bias_ih = self.bias_ih[ensemble_index]
            bias_hh = self.bias_hh[ensemble_index]
            for feature in sequence:
                input_gates = weight_ih @ feature + bias_ih
                hidden_gates = weight_hh @ hidden + bias_hh
                input_reset, input_update, input_new = np.split(input_gates, 3)
                hidden_reset, hidden_update, hidden_new = np.split(hidden_gates, 3)
                reset = _sigmoid(input_reset + hidden_reset)
                update = _sigmoid(input_update + hidden_update)
                candidate = np.tanh(input_new + reset * hidden_new)
                hidden = (1.0 - update) * candidate + update * hidden
            head = self.head0_weight[ensemble_index] @ hidden
            head += self.head0_bias[ensemble_index]
            head = head * _sigmoid(head)  # SiLU
            normalized = self.head2_weight[ensemble_index] @ head
            normalized += self.head2_bias[ensemble_index]
            predictions[ensemble_index] = (
                normalized * self.target_std + self.target_mean
            )
        return predictions.mean(axis=0), predictions.std(axis=0)
