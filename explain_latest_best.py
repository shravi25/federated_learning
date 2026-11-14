import sys
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_tabular import LimeTabularExplainer

from federated_app.task import Net
from federated_app.explainability import FEATURE_NAMES, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. AUTO-DETECT THE BEST CHECKPOINT
# =====================================================================
if len(sys.argv) > 1:
    # If the user passed a specific checkpoint path
    best_ckpt_path = sys.argv[1]
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    best_acc = best_ckpt.get("test_accuracy", 0.0)
else:
    # Scan all checkpoints to find the highest accuracy one
    print("[1/4] Scanning ./checkpoints/history/ for the highest-accuracy model...")
    all_ckpts = glob.glob("./checkpoints/history/*.pt")
    if not all_ckpts:
        raise FileNotFoundError("No checkpoints found in ./checkpoints/history/")

    best_ckpt_path, best_acc, best_ckpt = None, -1.0, None
    for p in all_ckpts:
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            acc = float(c.get("test_accuracy", 0.0))
            if acc > best_acc:
                best_acc = acc
                best_ckpt_path = p
                best_ckpt = c
        except Exception:
            continue

print("=" * 65)
print(f"SELECTED BEST MODEL CHECKPOINT:")
print(f"  File     : {best_ckpt_path}")
print(f"  Run ID   : {best_ckpt.get('run_id', 'Unknown')}")
print(f"  Round    : {best_ckpt.get('current_run_round', best_ckpt.get('cumulative_round', '?'))}")
print(f"  Accuracy : {best_acc * 100:.2f}%")
print("=" * 65 + "\n")

# Load model weights
model = Net().to(device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()

# =====================================================================
# 2. LOAD DATASET & PREDICTION FUNCTION
# =====================================================================
print("[2/4] Loading test dataset...")
data = torch.load("./datasets/covtype_preprocessed.pt", map_location="cpu", weights_only=False)
X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]

def predict_fn(x):
    model.eval()
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32).to(device)
        return F.softmax(model(t), dim=-1).cpu().numpy()

categorical_indices = [
    i for i, name in enumerate(FEATURE_NAMES)
    if name.startswith("Wilderness_") or name.startswith("Soil_Type_")
]

explainer = LimeTabularExplainer(
    training_data=X_train[:5000],
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    categorical_features=categorical_indices,
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# =====================================================================
# 3. LOCAL SAMPLE EXPLANATION (Sample #0)
# =====================================================================
print("[3/4] Generating LIME explanation for Sample #0...")
sample_idx = 0
instance = X_test[sample_idx]
true_label = int(y_test[sample_idx])

exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=5,
    top_labels=1
)
pred_label = exp.top_labels[0]
probs = predict_fn(instance[np.newaxis, :])[0]

print(f"\nSample #0 -> True: {CLASS_NAMES[true_label]} | Pred: {CLASS_NAMES[pred_label]} ({probs[pred_label]*100:.1f}% confidence)")
print("-" * 65)
for feat_cond, weight in exp.as_list(label=pred_label):
    direction = "Supports (+)" if weight > 0 else "Opposes (-)"
    print(f"  {feat_cond:<38} | {weight:+.4f} ({direction})")

# Save HTML report named after the run ID
run_id = best_ckpt.get("run_id", "best_model")
report_file = f"lime_report_{run_id}.html"
exp.save_to_file(report_file)
print(f"  -> Saved local report to: {report_file}\n")

# =====================================================================
# 4. GLOBAL LIME RANKING (Across 30 test samples)
# =====================================================================
print("[4/4] Calculating Global Feature Ranking across 30 samples...")
np.random.seed(42)
sampled_indices = np.random.choice(len(X_test), size=30, replace=False)
feature_weights = defaultdict(float)

for s_id in sampled_indices:
    sample_exp = explainer.explain_instance(
        data_row=X_test[s_id],
        predict_fn=predict_fn,
        num_features=5,
        top_labels=1
    )
    p_label = sample_exp.top_labels[0]
    for condition, w in sample_exp.as_list(label=p_label):
        base_col = next((col for col in FEATURE_NAMES if col in condition), condition)
        feature_weights[base_col] += abs(w)

total_w = sum(feature_weights.values())
sorted_impacts = sorted(
    [(feat, (w / total_w) * 100) for feat, w in feature_weights.items()],
    key=lambda x: x[1],
    reverse=True
)

print("=" * 65)
print(f"GLOBAL FEATURE IMPORTANCE FOR {run_id} ({best_acc*100:.2f}%)")
print("=" * 65)
for rank, (feat, pct) in enumerate(sorted_impacts[:8], 1):
    bar = "█" * int(pct // 2)
    print(f"#{rank:<2} {feat:<30} | {pct:6.2f}% | {bar}")
print("=" * 65)
