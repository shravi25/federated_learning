"""Model checkpointing and state persistence manager."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import torch
class CheckpointManager:
    """Manages persistent checkpoints, atomic saving, and experiment resumption."""
    def __init__(self, checkpoint_dir: Path | str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.history_dir = self.checkpoint_dir / "history"
        self.latest_path = self.checkpoint_dir / "latest_model.pt"
        self.best_path = self.checkpoint_dir / "best_model.pt"
        # Ensure target directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
    def load_for_resumption(
        self,
        auto_resume: bool = True,
        explicit_path: Path | str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Inspects and loads a model checkpoint to initialize or resume an experiment.
        Returns:
            checkpoint: Loaded checkpoint dict if available, otherwise None.
            resumption_info: Metadata dictionary describing initialization source.
        """
        target_path: Path | None = None
        if explicit_path:
            p = Path(explicit_path).resolve()
            if p.is_file():
                target_path = p
            else:
                raise FileNotFoundError(f"Explicit checkpoint not found: {p}")
        elif auto_resume and self.latest_path.is_file():
            target_path = self.latest_path
        if target_path is None:
            return None, {
                "source": "FRESH_INITIALIZATION",
                "checkpoint_path": None,
                "previous_run_id": None,
                "initial_cumulative_round": 0,
                "prior_best_accuracy": 0.0,
            }
        try:
            checkpoint = torch.load(
                target_path, map_location="cpu", weights_only=False
            )
            # Support both comprehensive dict format and raw state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                prev_run_id = checkpoint.get("run_id")
                cum_round = int(checkpoint.get("cumulative_round", 0))
                best_acc = float(
                    checkpoint.get(
                        "best_accuracy", checkpoint.get("test_accuracy", 0.0)
                    )
                )
            elif isinstance(checkpoint, dict):
                # Fallback for plain state_dict
                prev_run_id = None
                cum_round = 0
                best_acc = 0.0
                checkpoint = {
                    "model_state_dict": checkpoint,
                    "cumulative_round": cum_round,
                    "best_accuracy": best_acc,
                }
            else:
                raise ValueError(
                    f"Unrecognized checkpoint format at {target_path}"
                )
            return checkpoint, {
                "source": "RESUMED_FROM_CHECKPOINT",
                "checkpoint_path": str(target_path),
                "previous_run_id": prev_run_id,
                "initial_cumulative_round": cum_round,
                "prior_best_accuracy": best_acc,
            }
        except Exception as e:
            print(
                f"[CHECKPOINT WARNING] Could not load checkpoint {target_path}: {e}"
            )
            print(
                "[CHECKPOINT WARNING] Falling back to fresh model initialization."
            )
            return None, {
                "source": "FALLBACK_FRESH_INITIALIZATION",
                "checkpoint_path": None,
                "previous_run_id": None,
                "initial_cumulative_round": 0,
                "prior_best_accuracy": 0.0,
            }
    def _atomic_torch_save(
        self, data: dict[str, Any], target_path: Path
    ) -> None:
        """Saves PyTorch data atomically via temporary file and atomic rename."""
        temp_file = target_path.with_suffix(f".tmp_{os.getpid()}")
        try:
            torch.save(data, temp_file)
            os.replace(temp_file, target_path)
        except Exception:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise
    def save_checkpoint(
        self,
        model_state_dict: Mapping[str, torch.Tensor],
        cumulative_round: int,
        current_run_round: int,
        run_id: str,
        previous_run_id: str | None,
        test_loss: float,
        test_accuracy: float,
        best_accuracy: float,
        is_best: bool,
        model_name: str = "SimpleCNN",
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically saves checkpoint dictionary, updates latest and best models."""
        now_utc = datetime.now(timezone.utc).isoformat()
        # Clone state dict to CPU
        cpu_state_dict = {
            k: v.cpu().clone() for k, v in model_state_dict.items()
        }
        checkpoint_data: dict[str, Any] = {
            "model_state_dict": cpu_state_dict,
            "cumulative_round": cumulative_round,
            "current_run_round": current_run_round,
            "run_id": run_id,
            "previous_run_id": previous_run_id,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "best_accuracy": float(best_accuracy),
            "timestamp": now_utc,
            "model_name": model_name,
            "metadata": extra_metadata or {},
        }
        # 1. Update latest_model.pt
        self._atomic_torch_save(checkpoint_data, self.latest_path)
        # 2. Update best_model.pt if accuracy improved
        if is_best:
            self._atomic_torch_save(checkpoint_data, self.best_path)
        # 3. Archive snapshot in history
        history_path = (
            self.history_dir
            / f"checkpoint_run_{run_id}_round_{cumulative_round}.pt"
        )
        self._atomic_torch_save(checkpoint_data, history_path)
        return self.latest_path
