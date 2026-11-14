import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from lime.lime_tabular import LimeTabularExplainer

from federated_app.task import Net
from federated_app.explainability import FEATURE_NAMES, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
ckpt = torch.load("./checkpoints/history/checkpoint_run_run_20261110_092239_round_50.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

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

# Sample 30 random test instances
num_samples = 30
np.random.seed(42)
sampled_indices = np.random.choice(len(X_test), size=num_samples, replace=False)

print(f"Aggregating LIME explanations across {num_samples} test samples...")
feature_weights = defaultdict(float)

for idx, sample_id in enumerate(sampled_indices, 1):
    exp = explainer.explain_instance(
        data_row=X_test[sample_id],
        predict_fn=predict_fn,
        num_features=5,
        top_labels=1
    )
    pred_label = exp.top_labels[0]
    for condition, weight in exp.as_list(label=pred_label):
        # Extract base feature name
        base_feat = next((col for col in FEATURE_NAMES if col in condition), condition)
        feature_weights[base_feat] += abs(weight)
    
    if idx % 10 == 0:
        print(f"  Processed {idx}/{num_samples} samples...")

# Calculate percentage contribution
total_weight = sum(feature_weights.values())
sorted_impacts = sorted(
    [(feat, (w / total_weight) * 100) for feat, w in feature_weights.items()],
    key=lambda x: x[1],
    reverse=True
)

print("\n" + "=" * 60)
print(f"GLOBAL LIME FEATURE IMPACT RANKING ({num_samples} SAMPLES)")
print("=" * 60)
for rank, (feat, pct) in enumerate(sorted_impacts[:10], 1):
    bar = "█" * int(pct // 2)
    print(f"#{rank:<2} {feat:<30} | {pct:6.2f}% | {bar}")
print("=" * 60)
