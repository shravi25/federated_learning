"""Standalone script to run explainability analysis on saved FL model."""

from pathlib import Path
import torch

from federated_app.config import AppConfig
from federated_app.task import Net, load_centralized_test_data
from federated_app.explainability import ModelExplainer


def main():
    config = AppConfig()
    checkpoint_path = config.checkpoint_dir / "best_model.pt"

    if not checkpoint_path.exists():
        checkpoint_path = config.checkpoint_dir / "latest_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {config.checkpoint_dir}. Train the model first."
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = Net()
    model.load_state_dict(checkpoint["model_state_dict"])
    cumulative_round = checkpoint.get("cumulative_round", "N/A")
    print(f"Loaded global model at cumulative round: {cumulative_round}")

    # Load test set
    test_loader = load_centralized_test_data()

    # Explain model
    explainer = ModelExplainer(model)
    top_features = explainer.permutation_importance(test_loader, top_k=15)

    print("\n" + "=" * 50)
    print("TOP 15 MOST INFLUENTIAL FEATURES")
    print("=" * 50)
    for rank, (feat, drop) in enumerate(top_features, 1):
        print(f"{rank:2d}. {feat:<30} -> Impact: {drop * 100:.2f}% accuracy drop")
    print("=" * 50)

    # Save plot
    project_root = config.checkpoint_dir.parent
    output_plot = project_root / "plots" / "feature_importance.png"
    explainer.plot_global_importance(top_features, output_plot)


if __name__ == "__main__":
    main()