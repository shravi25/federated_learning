"""Configuration module for Persistent Federated Learning research."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Persistent project root anchored outside Flower's transient run execution sandbox
PROJECT_ROOT = Path.home() / "FL_Research" / "persistent-fl-research"


@dataclass
class AppConfig:
    """Experiment and system configuration."""

    num_server_rounds: int = 3
    num_clients: int = 2
    fraction_evaluate: float = 1.0
    local_epochs: int = 1
    learning_rate: float = 0.001
    batch_size: int = 128
    auto_resume: bool = False
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    results_dir: Path = PROJECT_ROOT / "results"
    data_dir: Path = PROJECT_ROOT / "datasets"
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_run_config(
        cls, run_config: Mapping[str, Any] | None = None
    ) -> "AppConfig":
        """Load configuration from Flower's run_config dictionary or defaults."""
        if not run_config:
            return cls()

        def _get(key: str, default: Any, cast_type: type):
            val = run_config.get(key, default)
            if val is None:
                return default
            if cast_type is bool and isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return cast_type(val)

        ckpt_dir_str = _get("checkpoint-dir", "checkpoints", str)
        ckpt_dir = Path(ckpt_dir_str)
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir

        res_dir_str = _get("results-dir", "results", str)
        res_dir = Path(res_dir_str)
        if not res_dir.is_absolute():
            res_dir = PROJECT_ROOT / res_dir

        data_dir_str = _get("data-dir", "datasets", str)
        data_dir = Path(data_dir_str)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

        return cls(
            num_server_rounds=_get("num-server-rounds", 3, int),
            num_clients=_get("num-clients", 2, int),
            fraction_evaluate=_get("fraction-evaluate", 1.0, float),
            local_epochs=_get("local-epochs", 1, int),
            learning_rate=_get("learning-rate", 0.001, float),
            batch_size=_get("batch-size", 128, int),
            auto_resume=_get("auto-resume", True, bool),
            checkpoint_dir=ckpt_dir,
            results_dir=res_dir,
            data_dir=data_dir,
            seed=_get("seed", 42, int),
            device="cpu",
        )
