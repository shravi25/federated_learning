"""
Standalone runner for SHAP explainability analysis.

Loads the trained federated model, computes SHAP values,
and saves global explanations.
"""

from pathlib import Path

import numpy as np
import torch

from federated_app.config import AppConfig
from federated_app.task import Net, load_covtype_data
from federated_app.explainability import CLASS_NAMES
from federated_app.shap_explainability import SHAPExplainer


def main():

    # --------------------------------------------------
    # 1. Load configuration and checkpoint
    # --------------------------------------------------

    config = AppConfig()

    checkpoint_path = config.checkpoint_dir / "best_model.pt"

    if not checkpoint_path.exists():
        checkpoint_path = config.checkpoint_dir / "latest_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No model checkpoint found in {config.checkpoint_dir}"
        )

    print("=" * 60)
    print("SHAP EXPLAINABILITY EXPERIMENT")
    print("=" * 60)

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = Net()

    model.load_state_dict(checkpoint["model_state_dict"])

    cumulative_round = checkpoint.get(
        "cumulative_round", "N/A"
    )

    print(
        f"Loaded model at cumulative round: {cumulative_round}"
    )

    # --------------------------------------------------
    # 2. Load Covertype dataset
    # --------------------------------------------------

    x_train, y_train, x_test, y_test = load_covtype_data()

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)

    print(f"Training data: {x_train.shape}")
    print(f"Testing data: {x_test.shape}")

    # --------------------------------------------------
    # 3. Select background and explanation samples
    # --------------------------------------------------

    rng = np.random.default_rng(42)

    background_indices = rng.choice(
        len(x_train),
        size=100,
        replace=False
    )

    test_indices = rng.choice(
        len(x_test),
        size=100,
        replace=False
    )

    background_data = x_train[background_indices]
    explanation_data = x_test[test_indices]

    print(f"Background samples: {len(background_data)}")
    print(f"Explanation samples: {len(explanation_data)}")

    # --------------------------------------------------
    # 4. Initialize SHAP
    # --------------------------------------------------

    explainer = SHAPExplainer(
        model=model,
        background_data=background_data
    )

    # --------------------------------------------------
    # 5. Calculate SHAP values
    # --------------------------------------------------

    shap_values = explainer.calculate_shap_values(
        explanation_data
    )

    # --------------------------------------------------
    # 6. Calculate global feature importance
    # --------------------------------------------------

    ranked_features, importance = explainer.global_importance(
        shap_values
    )

    print("\n" + "=" * 60)
    print("TOP 15 GLOBAL SHAP FEATURES")
    print("=" * 60)

    for rank, (feature, value) in enumerate(
        ranked_features[:15], start=1
    ):

        print(
            f"{rank:2d}. {feature:<30} "
            f"SHAP Value: {value:.6f}"
        )

    print("=" * 60)

    # --------------------------------------------------
    # 7. Prepare output directories
    # --------------------------------------------------

    project_root = config.checkpoint_dir.parent

    plots_dir = project_root / "plots"
    results_dir = project_root / "results"

    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 8. Save global importance plot
    # --------------------------------------------------

    global_plot = plots_dir / "shap_global_importance.png"

    explainer.plot_global_importance(
        ranked_features,
        global_plot,
        top_k=15
    )

    # --------------------------------------------------
    # 9. Save global SHAP summary plot
    # --------------------------------------------------

    summary_plot = plots_dir / "shap_summary_class_0.png"

    explainer.plot_summary(
        shap_values,
        explanation_data,
        summary_plot,
        class_index=0
    )

    # --------------------------------------------------
    # 10. Save numerical results
    # --------------------------------------------------

    csv_path = results_dir / "shap_global_importance.csv"

    explainer.save_global_results(
        ranked_features,
        csv_path
    )

    print("\nSHAP analysis completed successfully.")
    print(f"Global plot: {global_plot}")
    print(f"Summary plot: {summary_plot}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    main()