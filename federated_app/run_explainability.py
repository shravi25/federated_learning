"""Explainability module for Forest Cover Type Federated Learning model."""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 54 feature names corresponding to the Forest Cover Type dataset
FEATURE_NAMES: List[str] = [
    # 10 Continuous terrain features
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Dist_Hydrology",
    "Vertical_Dist_Hydrology",
    "Horizontal_Dist_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Dist_Fire_Points",
    # 4 Wilderness Area indicators
    "Wilderness_Rawah",
    "Wilderness_Neota",
    "Wilderness_Comanche_Peak",
    "Wilderness_Cache_la_Poudre",
] + [f"Soil_Type_{i}" for i in range(1, 41)]  # 40 Soil Types

CLASS_NAMES: List[str] = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


class ModelExplainer:
    """Computes Global and Local Feature Attributions for the FL Model."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()

    def permutation_importance(
        self,
        test_loader: DataLoader,
        top_k: int = 15,
    ) -> List[Tuple[str, float]]:
        """Compute Global Permutation Feature Importance (PFI).

        Measures the drop in classification accuracy when each feature's
        values are randomly permuted.
        """
        # Collect full test set into numpy arrays
        all_x, all_y = [], []
        for x_batch, y_batch in test_loader:
            all_x.append(x_batch.numpy())
            all_y.append(y_batch.numpy())
        x_all = np.concatenate(all_x, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        # Baseline accuracy
        baseline_acc = self._evaluate_accuracy(x_all, y_all)
        print(f"[XAI] Baseline Test Accuracy: {baseline_acc * 100:.2f}%")

        importances = {}
        rng = np.random.default_rng(42)

        for col_idx, feat_name in enumerate(FEATURE_NAMES):
            # Make a copy and shuffle only this feature column
            x_permuted = x_all.copy()
            x_permuted[:, col_idx] = rng.permutation(x_permuted[:, col_idx])

            # Measure performance drop
            permuted_acc = self._evaluate_accuracy(x_permuted, y_all)
            drop = max(0.0, baseline_acc - permuted_acc)
            importances[feat_name] = drop

        # Sort features by highest accuracy drop
        sorted_importances = sorted(
            importances.items(), key=lambda item: item[1], reverse=True
        )
        return sorted_importances[:top_k]

    def _evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Helper to evaluate accuracy on in-memory numpy data."""
        self.model.eval()
        correct, total = 0, len(y)
        batch_size = 512

        with torch.no_grad():
            for i in range(0, total, batch_size):
                bx = torch.from_numpy(x[i : i + batch_size]).to(self.device)
                by = torch.from_numpy(y[i : i + batch_size]).to(self.device)
                outputs = self.model(bx)
                preds = outputs.argmax(dim=-1)
                correct += (preds == by).sum().item()

        return correct / total

    def plot_global_importance(
        self,
        ranked_features: List[Tuple[str, float]],
        output_path: Path,
    ) -> None:
        """Generate and save horizontal bar chart of top features."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = [item[0] for item in reversed(ranked_features)]
        drops = [item[1] * 100 for item in reversed(ranked_features)]  # In percent

        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, drops, color="#1f77b4", edgecolor="black", alpha=0.85)
        plt.xlabel("Accuracy Drop When Shuffled (%)", fontsize=11, fontweight="bold")
        plt.title(
            "Global Feature Importance (Permutation Feature Importance)",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Label bars with exact drop percentages
        for bar, drop in zip(bars, drops):
            if drop > 0.05:
                plt.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{drop:.2f}%",
                    va="center",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[XAI] Feature importance plot saved to: {output_path}")