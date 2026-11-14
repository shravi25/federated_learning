"""Data loading, preprocessing, and neural network for Forest Cover Type classification."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path.home() / "FL_Research" / "persistent-fl-research"
DATA_PATH = PROJECT_DIR / "datasets" / "covtype.csv"
FALLBACK_DATA_PATH = Path.home() / "FL_Research" / "quickstart-pytorch" / "datasets" / "covtype.csv"

NUM_FEATURES = 54
NUM_CLASSES = 7
NUM_CONTINUOUS_FEATURES = 10
RANDOM_SEED = 42
TEST_SIZE = 0.2


class Net(nn.Module):
    """Deep neural network with BatchNorm and Dropout for Forest Cover Type."""

    def __init__(
        self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model() -> Net:
    """Instantiate a new Net model."""
    return Net()


# In-memory cached splits
_CACHED_SPLITS = None


def load_covtype_data():
    """Load, preprocess, and split the Forest Cover Type dataset."""
    global _CACHED_SPLITS
    if _CACHED_SPLITS is not None:
        return _CACHED_SPLITS

    target_path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Dataset not found at {target_path}.")

    df = pd.read_csv(target_path)
    if "Cover_Type" not in df.columns:
        raise ValueError("CSV must contain a 'Cover_Type' target column.")

    features = df.drop(columns=["Cover_Type"]).to_numpy(dtype=np.float32).copy()
    labels = df["Cover_Type"].to_numpy(dtype=np.int64) - 1

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Standardize continuous features only
    mean = x_train[:, :NUM_CONTINUOUS_FEATURES].mean(axis=0)
    std = x_train[:, :NUM_CONTINUOUS_FEATURES].std(axis=0)
    std[std == 0] = 1.0

    x_train[:, :NUM_CONTINUOUS_FEATURES] = (
        x_train[:, :NUM_CONTINUOUS_FEATURES] - mean
    ) / std
    x_test[:, :NUM_CONTINUOUS_FEATURES] = (
        x_test[:, :NUM_CONTINUOUS_FEATURES] - mean
    ) / std

    _CACHED_SPLITS = (x_train, y_train, x_test, y_test)
    return _CACHED_SPLITS


def get_class_weights(device: str = "cpu") -> torch.Tensor:
    """Calculate balanced class weights from the training split."""
    _, y_train, _, _ = load_covtype_data()
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_client_data(
    partition_id: int, num_partitions: int, batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for one simulated client."""
    x_train, y_train, _, _ = load_covtype_data()

    if num_partitions < 1 or not 0 <= partition_id < num_partitions:
        raise ValueError("Invalid partition_id or num_partitions.")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(x_train))
    client_partitions = np.array_split(indices, num_partitions)
    client_indices = client_partitions[partition_id]

    client_x = x_train[client_indices]
    client_y = y_train[client_indices]

    train_x, val_x, train_y, val_y = train_test_split(
        client_x,
        client_y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=client_y,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader


def load_centralized_test_data(batch_size: int = 256) -> DataLoader:
    """Create a DataLoader for the centralized held-out test set."""
    _, _, x_test, y_test = load_covtype_data()
    dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: str = "cpu",
    class_weights: torch.Tensor | None = None,
    max_batches: int | None = None,
) -> Tuple[float, float, int]:
    """Train the model locally with Cosine Annealing LR Scheduler and Weight Decay."""
    model.to(device)
    model.train()

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )
    
    # 1. Optimizer with L2 Weight Decay (prevents exploding weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 2. Learning Rate Scheduler: Smoothly decays LR from peak (lr) to 5% of peak
    total_steps = epochs * (
        min(len(trainloader), max_batches) if max_batches else len(trainloader)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.05
    )

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch_idx, (features, labels) in enumerate(trainloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            scheduler.step()  # <-- Step the scheduler after each batch!

            total_loss += loss.item() * features.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += features.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, total

def test(
    model: nn.Module,
    testloader: DataLoader,
    device: str = "cpu",
    class_weights: torch.Tensor | None = None,
    max_batches: int | None = None,
) -> Tuple[float, float, int]:
    """Evaluate model loss and accuracy."""
    model.to(device)
    model.eval()

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(testloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += features.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, total
