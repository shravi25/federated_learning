"""
SHAP explainability for the Federated Learning Covertype model.

This module provides:
1. Gradient-based SHAP values
2. Global feature importance
3. Global summary plot
4. Local explanation support
"""

from pathlib import Path

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

from federated_app.explainability import FEATURE_NAMES, CLASS_NAMES


class SHAPExplainer:

    def __init__(self, model, background_data, device=None):

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = model.to(self.device)
        self.model.eval()

        self.background_data = np.asarray(
            background_data, dtype=np.float32
        )

        self.background_tensor = torch.tensor(
            self.background_data, dtype=torch.float32
        ).to(self.device)

        print(f"SHAP device: {self.device}")
        print(f"Background samples: {len(self.background_data)}")

        # GradientExplainer uses the background dataset
        # as the reference distribution.
        self.explainer = shap.GradientExplainer(
            self.model,
            self.background_tensor
        )

    def calculate_shap_values(self, data):

        data = np.asarray(data, dtype=np.float32)

        data_tensor = torch.tensor(
            data, dtype=torch.float32
        ).to(self.device)

        print(f"Calculating SHAP values for {len(data)} samples...")

        shap_values = self.explainer.shap_values(data_tensor)

        # SHAP versions may return different formats.
        # Normalize output to a NumPy array.
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)

        shap_values = np.asarray(shap_values)

        print("SHAP values calculated.")
        print("SHAP shape:", shap_values.shape)

        return shap_values

    @staticmethod
    def normalize_shap_shape(shap_values, num_features, num_classes):

        """
        Normalize SHAP output into:
        (samples, features, classes)
        """

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 2:
            # Single output model
            shap_values = shap_values[:, :, np.newaxis]

        elif shap_values.ndim == 3:

            # Expected shape: (samples, features, classes)
            if (
                shap_values.shape[1] == num_features
                and shap_values.shape[2] == num_classes
            ):
                pass

            # Alternative shape: (samples, classes, features)
            elif (
                shap_values.shape[1] == num_classes
                and shap_values.shape[2] == num_features
            ):
                shap_values = np.transpose(
                    shap_values, (0, 2, 1)
                )

        else:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_values.shape}"
            )

        return shap_values

    def global_importance(self, shap_values):

        """
        Aggregate absolute SHAP values across:
        - samples
        - output classes

        Returns features ranked by mean absolute SHAP value.
        """

        shap_values = self.normalize_shap_shape(
            shap_values,
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASS_NAMES)
        )

        importance = np.mean(
            np.abs(shap_values),
            axis=(0, 2)
        )

        ranked_indices = np.argsort(importance)[::-1]

        ranked_features = [
            (FEATURE_NAMES[i], float(importance[i]))
            for i in ranked_indices
        ]

        return ranked_features, importance

    def plot_global_importance(
        self,
        ranked_features,
        output_path,
        top_k=15
    ):

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = [
            item[0] for item in ranked_features[:top_k]
        ]

        values = [
            item[1] for item in ranked_features[:top_k]
        ]

        features = features[::-1]
        values = values[::-1]

        plt.figure(figsize=(10, 8))

        plt.barh(features, values)

        plt.xlabel("Mean Absolute SHAP Value")
        plt.ylabel("Feature")
        plt.title(f"Top {top_k} Global SHAP Features")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Global SHAP plot saved: {output_path}")

    def plot_summary(
        self,
        shap_values,
        explanation_data,
        output_path,
        class_index=0
    ):

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap_values = self.normalize_shap_shape(
            shap_values,
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASS_NAMES)
        )

        selected_shap = shap_values[:, :, class_index]

        plt.figure()

        shap.summary_plot(
            selected_shap,
            explanation_data,
            feature_names=FEATURE_NAMES,
            show=False
        )

        plt.title(
            f"SHAP Summary - {CLASS_NAMES[class_index]}"
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"SHAP summary plot saved: {output_path}")

    def save_global_results(
        self,
        ranked_features,
        output_path
    ):

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as file:

            file.write(
                "Rank,Feature,Mean Absolute SHAP Value\n"
            )

            for rank, (feature, value) in enumerate(
                ranked_features, start=1
            ):

                file.write(
                    f"{rank},{feature},{value}\n"
                )

        print(f"SHAP results saved: {output_path}")