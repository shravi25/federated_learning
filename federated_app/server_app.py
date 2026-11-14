"""Flower ServerApp for Forest Cover Type federated learning with persistence."""
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from federated_app.run_explainability import ModelExplainer
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from federated_app.checkpoint import CheckpointManager
from federated_app.config import AppConfig
from federated_app.metrics import ExperimentTracker
from federated_app.task import (
    get_model,
    load_centralized_test_data,
    get_class_weights,
    test,
)
app = ServerApp()
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run persistent federated learning experiment."""
    # 1. Load experiment configuration
    cfg = AppConfig.from_run_config(context.run_config)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    experiment_id = str(context.run_config.get("experiment-id", "forest_cover_fl"))
    # 2. Initialize persistence engine and experiment tracker
    checkpoint_mgr = CheckpointManager(cfg.checkpoint_dir)
    tracker = ExperimentTracker(cfg.results_dir)
    print("\n" + "=" * 66)
    print("      PERSISTENT FEDERATED LEARNING - SERVER INITIALIZING")
    print("=" * 66)
    print(f"Run ID                 : {run_id}")
    print(f"Experiment ID          : {experiment_id}")
    print(f"Rounds this run        : {cfg.num_server_rounds}")
    print(f"Configured clients     : {cfg.num_clients}")
    print(f"Learning rate          : {cfg.learning_rate}")
    print(f"Batch size             : {cfg.batch_size}")
    print(f"Auto-resumption active : {cfg.auto_resume}")
    print(f"Checkpoint directory   : {cfg.checkpoint_dir}")
    print(f"Results directory      : {cfg.results_dir}")
    # 3. Model Resumption or Fresh Initialization
    global_model = get_model()
    checkpoint, resumption_info = checkpoint_mgr.load_for_resumption(
        auto_resume=cfg.auto_resume
    )
    init_source = resumption_info["source"]
    start_cumulative_round = resumption_info["initial_cumulative_round"]
    previous_run_id = resumption_info["previous_run_id"]
    best_accuracy = float(resumption_info["prior_best_accuracy"])
    if checkpoint is not None:
        global_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[MODEL INITIALIZATION: RESUMED]")
        print(f"  Source checkpoint     : {resumption_info['checkpoint_path']}")
        print(f"  Previous Run ID       : {previous_run_id}")
        print(f"  Resumed round offset  : {start_cumulative_round}")
        print(f"  Prior best accuracy   : {best_accuracy * 100:.2f}%")
        print(f"  >> SUCCESS: Loaded learned weights from previous experiment!")
    else:
        print(f"\n[MODEL INITIALIZATION: FRESH START]")
        print(f"  No previous checkpoint found.")
        print(f"  Initialized fresh model with seed {cfg.seed}.")
        print(f"  Starting from cumulative round 0.")
    # 4. Centralized test set & pre-training baseline evaluation
    testloader = load_centralized_test_data(batch_size=256)
    class_weights = get_class_weights(device="cpu")
    base_loss, base_acc, _ = test(global_model, testloader, device="cpu", class_weights=class_weights)
    print(f"\n[PRE-TRAIN BASELINE] Test Loss: {base_loss:.4f} | Test Acc: {base_acc*100:.2f}%")
    if base_acc > best_accuracy:
        best_accuracy = base_acc
    initial_arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvg(fraction_evaluate=cfg.fraction_evaluate)
    round_start_time = time.perf_counter()
    # 5. Define evaluation & persistent checkpointing callback per communication round
    def evaluate_and_checkpoint(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:
        nonlocal best_accuracy, round_start_time
        cumulative_round = start_cumulative_round + server_round
        round_duration = time.perf_counter() - round_start_time
        eval_model = get_model()
        eval_model.load_state_dict(arrays.to_torch_state_dict())
        # Centralized test evaluation
        test_loss, test_acc, _ = test(eval_model, testloader, device="cpu", class_weights=class_weights)
        is_best = test_acc > best_accuracy
        if is_best:
            best_accuracy = test_acc
        # Atomically save latest and best model checkpoints
        saved_path = checkpoint_mgr.save_checkpoint(
            model_state_dict=eval_model.state_dict(),
            cumulative_round=cumulative_round,
            current_run_round=server_round,
            run_id=run_id,
            previous_run_id=previous_run_id,
            test_loss=test_loss,
            test_accuracy=test_acc,
            best_accuracy=best_accuracy,
            is_best=is_best,
            model_name="Net",
        )
        # Append round metrics to CSV
        tracker.log_round(
            experiment_id=experiment_id,
            run_id=run_id,
            cumulative_round=cumulative_round,
            run_round=server_round,
            client_participation=cfg.num_clients,
            train_loss=0.0,
            val_loss=0.0,
            val_accuracy=0.0,
            global_test_loss=test_loss,
            global_test_accuracy=test_acc,
            round_duration_sec=round_duration,
            checkpoint_path=str(saved_path),
            init_source=init_source,
        )
        status_flag = " [NEW BEST MODEL!]" if is_best else ""
        print(
            f"\n>>> [Round {server_round}/{cfg.num_server_rounds} | Cumulative Round {cumulative_round}]"
            f" Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}% | Best: {best_accuracy*100:.2f}%{status_flag}"
        )
        print(f"    Checkpoint saved -> {saved_path.name}")
        round_start_time = time.perf_counter()
        return MetricRecord({"loss": float(test_loss), "accuracy": float(test_acc)})
    # 6. Execute federated aggregation
    total_start_time = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=cfg.num_server_rounds,
        train_config=ConfigRecord({"lr": cfg.learning_rate}),
        evaluate_fn=evaluate_and_checkpoint,
    )
    total_runtime = time.perf_counter() - total_start_time
    end_cumulative_round = start_cumulative_round + cfg.num_server_rounds
    # 7. Final Model Evaluation & Summary Logging
        # 7. Final Model Evaluation
    final_model = get_model()
    final_model.load_state_dict(result.arrays.to_torch_state_dict())
    final_loss, final_acc, _ = test(
        final_model, testloader, device="cpu", class_weights=class_weights
    )

    # 8. Explainability: Feature Importance Analysis
    from federated_app.explainability import ModelExplainer

    explainer = ModelExplainer(final_model)
    top_features = explainer.permutation_importance(testloader, top_k=10)
    explainer.plot_global_importance(
        top_features, cfg.results_dir.parent / "plots" / "feature_importance.png"
    )

    # 9. Summary Logging (Includes Hyperparameters + Explainability)
    tracker.log_experiment_summary(
        run_id=run_id,
        rounds_executed=cfg.num_server_rounds,
        num_clients=cfg.num_clients,
        local_epochs=cfg.local_epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        initial_test_loss=base_loss,
        final_test_loss=final_loss,
        initial_test_accuracy=base_acc,
        final_test_accuracy=final_acc,
        best_test_accuracy=best_accuracy,
        total_runtime_sec=total_runtime,
        checkpoint_saved=str(checkpoint_mgr.latest_path),
        top_features=top_features,  # <-- Pass top_features here!
    )
    # Generate updated accuracy and loss plots
    acc_chart, loss_chart = tracker.generate_plots()
    print("\n" + "=" * 66)
    print("           FEDERATED LEARNING RUN COMPLETED")
    print("=" * 66)
    print(f"Run ID                : {run_id}")
    print(f"Cumulative rounds     : {start_cumulative_round} -> {end_cumulative_round}")
    print(f"Baseline accuracy     : {base_acc*100:.2f}%")
    print(f"Final round accuracy  : {final_acc*100:.2f}%")
    print(f"Best accuracy so far  : {best_accuracy*100:.2f}%")
    print(f"Total run time        : {total_runtime:.2f} seconds")
    print(f"Persistent checkpoint : {checkpoint_mgr.latest_path}")
    print(f"Metrics CSV           : {tracker.metrics_csv}")
    print(f"History CSV           : {tracker.history_csv}")
    print(f"Accuracy plot         : {acc_chart}")
    print(f"Loss plot             : {loss_chart}")
    print("=" * 66 + "\n")

    # At the end of server_app.py (inside @app.main() after training loop finishes):
    from federated_app.explainability import ModelExplainer

    explainer = ModelExplainer(final_model)
    top_features = explainer.permutation_importance(testloader, top_k=10)
    explainer.plot_global_importance(
        top_features, cfg.results_dir.parent / "plots" / "feature_importance.png"
    )