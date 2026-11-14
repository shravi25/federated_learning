"""Experiment tracking, CSV logging, and visualization for federated learning."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROUND_METRIC_FIELDS = [
    "experiment_id",
    "run_id",
    "cumulative_round",
    "run_round",
    "client_participation",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "global_test_loss",
    "global_test_accuracy",
    "round_duration_sec",
    "checkpoint_path",
    "init_source",
    "timestamp",
]

EXPERIMENT_HISTORY_FIELDS = [
    "run_id",
    "num_clients",
    "rounds_executed",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "initial_test_loss",
    "final_test_loss",
    "initial_test_accuracy",
    "final_test_accuracy",
    "best_test_accuracy",
    "total_runtime_sec",
    "top_1_feature",
    "top_1_impact_pct",
    "top_2_feature",
    "top_2_impact_pct",
    "top_3_feature",
    "top_3_impact_pct",
    "top_features_summary",
    "init_source",
    "timestamp",
]

FEATURE_IMPORTANCE_FIELDS = [
    "run_id",
    "rank",
    "feature_name",
    "accuracy_drop_pct",
    "timestamp",
]


class ExperimentTracker:
    """Manages persistent CSV logging and experiment visualization."""

    def __init__(
        self, results_dir: Path | str, plots_dir: Path | str | None = None
    ) -> None:
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = (
            Path(plots_dir).resolve()
            if plots_dir
            else self.results_dir.parent / "plots"
        )
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_csv = self.results_dir / "metrics.csv"
        self.history_csv = self.results_dir / "experiment_history.csv"
        self.feature_importance_csv = self.results_dir / "feature_importance.csv"

        self._ensure_header(self.metrics_csv, ROUND_METRIC_FIELDS)
        self._ensure_header(self.history_csv, EXPERIMENT_HISTORY_FIELDS)
        self._ensure_header(self.feature_importance_csv, FEATURE_IMPORTANCE_FIELDS)

    def _ensure_header(self, file_path: Path, fieldnames: list[str]) -> None:
        """Create CSV file with header if it does not exist or is empty."""
        if not file_path.exists() or file_path.stat().st_size == 0:
            with file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log_round(
        self,
        experiment_id: str,
        run_id: str,
        cumulative_round: int,
        run_round: int,
        client_participation: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        global_test_loss: float,
        global_test_accuracy: float,
        round_duration_sec: float,
        checkpoint_path: str,
        init_source: str,
    ) -> None:
        """Append round results to metrics.csv."""
        row = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "cumulative_round": cumulative_round,
            "run_round": run_round,
            "client_participation": client_participation,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_accuracy": round(float(val_accuracy), 6),
            "global_test_loss": round(float(global_test_loss), 6),
            "global_test_accuracy": round(float(global_test_accuracy), 6),
            "round_duration_sec": round(float(round_duration_sec), 2),
            "checkpoint_path": checkpoint_path,
            "init_source": init_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ROUND_METRIC_FIELDS)
            writer.writerow(row)

    def log_feature_importance(
        self, run_id: str, ranked_features: List[Tuple[str, float]]
    ) -> None:
        """Append detailed feature importance rankings to feature_importance.csv."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self.feature_importance_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_IMPORTANCE_FIELDS)
            for rank, (feat_name, drop) in enumerate(ranked_features, 1):
                writer.writerow(
                    {
                        "run_id": run_id,
                        "rank": rank,
                        "feature_name": feat_name,
                        "accuracy_drop_pct": round(float(drop) * 100, 2),
                        "timestamp": now_ts,
                    }
                )

    def log_experiment_summary(
        self,
        run_id: str,
        num_clients: int = 2,
        rounds_executed: int = 5,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        initial_test_loss: float = 0.0,
        final_test_loss: float = 0.0,
        initial_test_accuracy: float = 0.0,
        final_test_accuracy: float = 0.0,
        best_test_accuracy: float = 0.0,
        total_runtime_sec: float = 0.0,
        init_source: str = "FRESH_INITIALIZATION",
        checkpoint_saved: str = "",
        top_features: List[Tuple[str, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Append run summary with hyperparameters and explainability to experiment_history.csv."""
        # Extract top 3 features and summary string
        top_1_name = top_features[0][0] if top_features and len(top_features) > 0 else "N/A"
        top_1_pct = round(float(top_features[0][1]) * 100, 2) if top_features and len(top_features) > 0 else 0.0

        top_2_name = top_features[1][0] if top_features and len(top_features) > 1 else "N/A"
        top_2_pct = round(float(top_features[1][1]) * 100, 2) if top_features and len(top_features) > 1 else 0.0

        top_3_name = top_features[2][0] if top_features and len(top_features) > 2 else "N/A"
        top_3_pct = round(float(top_features[2][1]) * 100, 2) if top_features and len(top_features) > 2 else 0.0

        summary_str = (
            "; ".join([f"{feat} ({drop*100:.1f}%)" for feat, drop in top_features[:5]])
            if top_features
            else "N/A"
        )

        row = {
            "run_id": run_id,
            "num_clients": int(num_clients),
            "rounds_executed": int(rounds_executed),
            "local_epochs": int(local_epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "initial_test_loss": round(float(initial_test_loss), 4),
            "final_test_loss": round(float(final_test_loss), 4),
            "initial_test_accuracy": round(float(initial_test_accuracy), 4),
            "final_test_accuracy": round(float(final_test_accuracy), 4),
            "best_test_accuracy": round(float(best_test_accuracy), 4),
            "total_runtime_sec": round(float(total_runtime_sec), 2),
            "top_1_feature": top_1_name,
            "top_1_impact_pct": top_1_pct,
            "top_2_feature": top_2_name,
            "top_2_impact_pct": top_2_pct,
            "top_3_feature": top_3_name,
            "top_3_impact_pct": top_3_pct,
            "top_features_summary": summary_str,
            "init_source": str(init_source),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.history_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EXPERIMENT_HISTORY_FIELDS)
            writer.writerow(row)

        # Also write the full detailed table into feature_importance.csv
        if top_features:
            self.log_feature_importance(run_id, top_features)

    def generate_plots(self) -> tuple[Path, Path]:
        """Generate and save accuracy and loss charts."""
        if not self.metrics_csv.exists() or self.metrics_csv.stat().st_size == 0:
            return self.plots_dir / "accuracy_vs_round.png", self.plots_dir / "loss_vs_round.png"

        df = pd.read_csv(self.metrics_csv)
        if df.empty or "cumulative_round" not in df.columns:
            return self.plots_dir / "accuracy_vs_round.png", self.plots_dir / "loss_vs_round.png"

        plt.figure(figsize=(9, 5))
        plt.plot(df["cumulative_round"], df["global_test_accuracy"] * 100, marker="o", linewidth=2, color="#1f77b4", label="Global Test Accuracy (%)")
        plt.title("Federated Learning Accuracy vs. Communication Round")
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        acc_plot_path = self.plots_dir / "accuracy_vs_round.png"
        plt.savefig(acc_plot_path, dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(df["cumulative_round"], df["global_test_loss"], marker="o", linewidth=2, color="#d62728", label="Global Test Loss")
        plt.title("Federated Learning Loss vs. Communication Round")
        plt.xlabel("Communication Round")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        loss_plot_path = self.plots_dir / "loss_vs_round.png"
        plt.savefig(loss_plot_path, dpi=180)
        plt.close()

        return acc_plot_path, loss_plot_path
