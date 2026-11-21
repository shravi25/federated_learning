"""Standalone script to run PFI and LIME explainability on saved FL model."""

from pathlib import Path
import torch
import numpy as np

from federated_app.config import AppConfig
from federated_app.task import (
    Net,
    load_centralized_test_data,
    load_covtype_data,
)
from federated_app.explainability import ModelExplainer


def to_numpy(data):
    """Convert data to NumPy if it is a Torch tensor."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def main():
    config = AppConfig()

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint_path = config.checkpoint_dir / "best_model.pt"

    if not checkpoint_path.exists():
        checkpoint_path = config.checkpoint_dir / "latest_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {config.checkpoint_dir}. Train the model first."
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = Net()
    model.load_state_dict(checkpoint["model_state_dict"])

    cumulative_round = checkpoint.get("cumulative_round", "N/A")

    print(
        f"Loaded global model at cumulative round: "
        f"{cumulative_round}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    x_train, y_train, x_test, y_test = load_covtype_data()

    x_train = to_numpy(x_train).astype(np.float32)
    x_test = to_numpy(x_test).astype(np.float32)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    test_loader = load_centralized_test_data()

    # ========================================================
    # INITIALIZE EXPLAINER
    # ========================================================

    explainer = ModelExplainer(model)

    # ========================================================
    # GLOBAL EXPLAINABILITY - PFI
    # ========================================================

    print("\nRunning Permutation Feature Importance...")

    top_features = explainer.permutation_importance(
        test_loader,
        top_k=15,
    )

    print("\n" + "=" * 50)
    print("TOP 15 MOST INFLUENTIAL FEATURES")
    print("=" * 50)

    for rank, (feat, drop) in enumerate(top_features, 1):
        print(
            f"{rank:2d}. {feat:<30} "
            f"-> Impact: {drop * 100:.2f}% accuracy drop"
        )

    print("=" * 50)

    # Save PFI plot
    project_root = config.checkpoint_dir.parent
    output_plot = (
        project_root
        / "plots"
        / "feature_importance.png"
    )

    explainer.plot_global_importance(
        top_features,
        output_plot,
    )

    # ========================================================
    # LOCAL EXPLAINABILITY - LIME
    # ========================================================

    print("\nInitializing LIME...")

    explainer.initialize_lime(x_train)

    # Explain one test sample
    sample_index = 1

    print(
        f"\nGenerating LIME explanation "
        f"for test sample {sample_index}..."
    )

    explanation, predicted_class, probabilities = (
        explainer.explain_instance(
            test_data=x_test,
            sample_index=sample_index,
            num_features=10,
            num_samples=5000,
        )
    )

    # ========================================================
    # SAVE LIME OUTPUTS
    # ========================================================

    lime_plot = (
        project_root
        / "plots"
        / f"lime_sample_{sample_index}.png"
    )

    lime_csv = (
        project_root
        / "results"
        / f"lime_sample_{sample_index}.csv"
    )

    explainer.plot_lime_explanation(
        explanation,
        predicted_class,
        lime_plot,
    )

    explainer.save_lime_explanation(
        explanation,
        predicted_class,
        lime_csv,
    )

    print("\n" + "=" * 50)
    print("EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 50)

    print(f"PFI plot: {output_plot}")
    print(f"LIME plot: {lime_plot}")
    print(f"LIME CSV: {lime_csv}")
    print(f"Explained sample: {sample_index}")
    print(f"Predicted class: {predicted_class}")
    print(f"Predicted label: {explainer.lime_explainer.class_names[predicted_class]}")
    print(
        f"Prediction confidence: "
        f"{probabilities[predicted_class] * 100:.2f}%"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()