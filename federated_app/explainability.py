"""Explainability module for Forest Cover Type Federated Learning model."""

from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# FEATURE NAMES
# ============================================================

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
] + [f"Soil_Type_{i}" for i in range(1, 41)]


CLASS_NAMES: List[str] = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


# ============================================================
# MODEL EXPLAINER
# ============================================================

class ModelExplainer:
    """Computes Global and Local Feature Attributions for the FL Model."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.lime_explainer = None


    # ========================================================
    # GLOBAL EXPLAINABILITY - PERMUTATION FEATURE IMPORTANCE
    # ========================================================

    def permutation_importance(
        self,
        test_loader: DataLoader,
        top_k: int = 10,
        max_samples: int = 10000,
    ) -> List[Tuple[str, float]]:

        all_x, all_y = [], []
        total_collected = 0

        for x_batch, y_batch in test_loader:
            all_x.append(x_batch.numpy())
            all_y.append(y_batch.numpy())

            total_collected += len(x_batch)

            if total_collected >= max_samples:
                break

        x_all = np.concatenate(all_x, axis=0)[:max_samples]
        y_all = np.concatenate(all_y, axis=0)[:max_samples]

        print(
            f"\n[XAI] Calculating Permutation Feature Importance "
            f"on {len(x_all)} test samples..."
        )

        baseline_acc = self._evaluate_accuracy(x_all, y_all)

        print(f"[XAI] Baseline Test Accuracy: {baseline_acc * 100:.2f}%")

        importances = {}
        rng = np.random.default_rng(42)

        for col_idx, feat_name in enumerate(FEATURE_NAMES):

            x_permuted = x_all.copy()

            x_permuted[:, col_idx] = rng.permutation(
                x_permuted[:, col_idx]
            )

            permuted_acc = self._evaluate_accuracy(
                x_permuted, y_all
            )

            drop = max(0.0, baseline_acc - permuted_acc)

            importances[feat_name] = drop

        sorted_importances = sorted(
            importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        print("\n" + "=" * 55)
        print("  TOP INFLUENTIAL FEATURES")
        print("=" * 55)

        for rank, (feat, drop) in enumerate(
            sorted_importances[:top_k], 1
        ):
            print(
                f"{rank:2d}. {feat:<28} "
                f"-> {drop * 100:.2f}% drop"
            )

        print("=" * 55)

        return sorted_importances[:top_k]


    def _evaluate_accuracy(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:

        self.model.eval()

        correct = 0
        total = len(y)
        batch_size = 512

        with torch.no_grad():

            for i in range(0, total, batch_size):

                bx = torch.from_numpy(
                    x[i:i + batch_size]
                ).float().to(self.device)

                by = torch.from_numpy(
                    y[i:i + batch_size]
                ).to(self.device)

                outputs = self.model(bx)

                preds = outputs.argmax(dim=-1)

                correct += (preds == by).sum().item()

        return correct / total


    def plot_global_importance(
        self,
        ranked_features: List[Tuple[str, float]],
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        features = [
            item[0] for item in reversed(ranked_features)
        ]

        drops = [
            item[1] * 100 for item in reversed(ranked_features)
        ]

        plt.figure(figsize=(10, 6))

        bars = plt.barh(
            features,
            drops,
            color="#1f77b4",
            edgecolor="black",
            alpha=0.85,
        )

        plt.xlabel(
            "Accuracy Drop When Scrambled (%)",
            fontsize=11,
            fontweight="bold",
        )

        plt.title(
            "Global Feature Importance "
            "(Permutation Feature Importance)",
            fontsize=12,
            fontweight="bold",
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.7,
        )

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

        plt.savefig(
            output_path,
            dpi=300,
        )

        plt.close()

        print(
            f"[XAI] Feature importance plot saved to: "
            f"{output_path}"
        )


    # ========================================================
    # LOCAL EXPLAINABILITY - LIME
    # ========================================================

    def initialize_lime(
        self,
        training_data: np.ndarray,
    ) -> None:
        """Initialize LIME using training data as reference."""

        print("\n[XAI] Initializing LIME...")

        # Binary indicator features: wilderness + soil
        categorical_features = list(range(10, 54))

        categorical_names = {}

        for index in categorical_features:

            categorical_names[index] = [
                "Absent",
                "Present",
            ]

        self.lime_explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=FEATURE_NAMES,
            class_names=CLASS_NAMES,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        print("[XAI] LIME initialized successfully.")


    def predict_proba(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Convert model logits into probabilities for LIME."""

        self.model.eval()

        probabilities = []

        batch_size = 512

        with torch.no_grad():

            for i in range(0, len(data), batch_size):

                batch = torch.tensor(
                    data[i:i + batch_size],
                    dtype=torch.float32,
                ).to(self.device)

                logits = self.model(batch)

                probs = torch.softmax(
                    logits,
                    dim=1,
                )

                probabilities.append(
                    probs.cpu().numpy()
                )

        return np.concatenate(
            probabilities,
            axis=0,
        )


    def explain_instance(
        self,
        test_data: np.ndarray,
        sample_index: int,
        num_features: int = 10,
        num_samples: int = 5000,
    ):
        """Generate a local LIME explanation for a test sample."""

        if self.lime_explainer is None:

            raise RuntimeError(
                "LIME has not been initialized. "
                "Call initialize_lime() first."
            )

        if sample_index < 0 or sample_index >= len(test_data):

            raise IndexError(
                "Sample index is outside the test dataset."
            )

        sample = test_data[sample_index]

        probabilities = self.predict_proba(
            sample.reshape(1, -1)
        )

        predicted_class = int(
            np.argmax(probabilities[0])
        )

        print("\n" + "=" * 55)
        print("[XAI] LOCAL LIME EXPLANATION")
        print("=" * 55)

        print(
            f"Sample index: {sample_index}"
        )

        print(
            f"Predicted class: "
            f"{CLASS_NAMES[predicted_class]}"
        )

        print(
            f"Prediction confidence: "
            f"{probabilities[0][predicted_class] * 100:.2f}%"
        )

        explanation = self.lime_explainer.explain_instance(
            data_row=sample,
            predict_fn=self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1,
        )

        print("\nFeature contributions:")

        for feature, weight in explanation.as_list(
            label=predicted_class
        ):

            print(
                f"{feature:<35} {weight:+.4f}"
            )

        print("=" * 55)

        return explanation, predicted_class, probabilities[0]


    def plot_lime_explanation(
        self,
        explanation,
        predicted_class: int,
        output_path: Path,
    ) -> None:
        """Save LIME local explanation visualization."""

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig = explanation.as_pyplot_figure(
            label=predicted_class
        )

        fig.set_size_inches(
            10,
            6,
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"[XAI] LIME explanation saved to: "
            f"{output_path}"
        )


    def save_lime_explanation(
        self,
        explanation,
        predicted_class: int,
        output_path: Path,
    ) -> None:
        """Save LIME feature contributions as CSV."""

        import csv

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        contributions = explanation.as_list(
            label=predicted_class
        )

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Feature",
                "Contribution",
            ])

            writer.writerows(
                contributions
            )

        print(
            f"[XAI] LIME results saved to: "
            f"{output_path}"
        )