"""TorchScript runtime for residual attention-GRU ensembles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TorchScriptResidualGRUEnsemble:
    """Load deployable PyTorch residual models behind the NumPy runtime contract."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on launch environment
            raise RuntimeError(
                "PyTorch is required for residual_model_backend=torchscript"
            ) from exc

        path = Path(model_path).expanduser().resolve()
        manifest_path = path / "torchscript_ensemble.json" if path.is_dir() else path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"TorchScript ensemble manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("backend") != "torchscript":
            raise ValueError(f"Unsupported residual ensemble backend: {manifest.get('backend')}")
        root = manifest_path.parent
        metadata_path = root / str(manifest.get("metadata", "model_metadata.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset = metadata["dataset"]

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA residual inference requested but CUDA is unavailable")
        if self.device.type == "cpu":
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        self.models = []
        for name in manifest.get("models", []):
            model = torch.jit.load(str(root / str(name)), map_location=self.device)
            self.models.append(model.eval())
        if not self.models:
            raise ValueError("TorchScript residual ensemble contains no models")

        self.history_samples = int(metadata["history_samples"])
        self.rate_hz = float(dataset["rate_hz"])
        self.ensemble_size = len(self.models)
        self.input_size = len(dataset["feature_names"])
        self.output_size = len(dataset["target_names"])
        self.model_type = str(metadata.get("model_type", "unknown"))

    def predict(self, raw_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence = np.asarray(raw_sequence, dtype=np.float32)
        expected = (self.history_samples, self.input_size)
        if sequence.shape != expected:
            raise ValueError(f"Expected sequence {expected}, received {sequence.shape}")
        tensor = self.torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            predictions = self.torch.stack(
                [model(tensor).squeeze(0) for model in self.models], dim=0
            )
            mean = predictions.mean(dim=0).cpu().numpy().astype(np.float64)
            std = predictions.std(dim=0, unbiased=False).cpu().numpy().astype(np.float64)
        return mean, std
