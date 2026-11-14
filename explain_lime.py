import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_tabular import LimeTabularExplainer

from federated_app.task import Net
from federated_app.explainability import FEATURE_NAMES, CLASS_NAMES

print("[1/5] Loading Peak-Accuracy Model Checkpoint...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

ckpt_path = "./checkpoints/history/checkpoint_run_run_20261110_092239_round_50.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"      Loaded: {ckpt_path}")
print(f"      Model Test Accuracy: {ckpt.get('test_accuracy', 0)*100:.2f}%\n")

print("[2/5] Loading Preprocessed Dataset...")
data = torch.load("./datasets/covtype_preprocessed.pt", map_location="cpu", weights_only=False)
X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
print(f"      X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\n")

# Use a representative subset of 5,000 background samples for fast LIME calibration
background_data = X_train[:5000]

print("[3/5] Defining PyTorch Prediction Wrapper...")
def predict_fn(numpy_inputs: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensors = torch.tensor(numpy_inputs, dtype=torch.float32).to(device)
        logits = model(tensors)
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

print("[4/5] Initializing LimeTabularExplainer...")
categorical_indices = [
    i for i, name in enumerate(FEATURE_NAMES)
    if name.startswith("Wilderness_") or name.startswith("Soil_Type_")
]

explainer = LimeTabularExplainer(
    training_data=background_data,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    categorical_features=categorical_indices,
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

print("[5/5] Generating LIME Explanation for Test Sample #0...")
sample_idx = 0
instance = X_test[sample_idx]
true_label = int(y_test[sample_idx])

# Run LIME
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=5,
    top_labels=1
)

predicted_label = exp.top_labels[0]
probs = predict_fn(instance[np.newaxis, :])[0]

print("\n" + "=" * 65)
print(f"SAMPLE #{sample_idx} RESULTS:")
print(f"  Ground Truth Class : {CLASS_NAMES[true_label]} (ID: {true_label})")
print(f"  Predicted Class    : {CLASS_NAMES[predicted_label]} (ID: {predicted_label})")
print(f"  Confidence         : {probs[predicted_label] * 100:.2f}%")
print("=" * 65)
print(f"{'Feature Condition':<40} | {'Weight':<10} | Contribution")
print("-" * 65)

for feat_condition, weight in exp.as_list(label=predicted_label):
    direction = "Supports (+)" if weight > 0 else "Opposes (-)"
    print(f"{feat_condition:<40} | {weight:+.4f}    | {direction}")

print("=" * 65)

# Save interactive visualization report
html_file = "lime_report.html"
exp.save_to_file(html_file)
print(f"\n[OK] Interactive visual explanation saved to: {html_file}\n")
