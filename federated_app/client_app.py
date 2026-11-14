"""Flower ClientApp for Forest Cover Type federated learning."""
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from federated_app.task import (
    get_model,
    load_client_data,
    get_class_weights,
    train,
    test,
)
app = ClientApp()
@app.train()
def train_client(msg: Message, context: Context) -> Message:
    """Train the model on the local client partition."""
    # 1. Initialize architecture and load current global parameters
    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(context.run_config.get("device", "cpu"))
    # 2. Extract partition configurations with robust fallbacks
    partition_id = int(context.node_config.get("partition-id", 0))
    num_partitions = int(
        context.node_config.get(
            "num-partitions", context.run_config.get("num-clients", 2)
        )
    )
    batch_size = int(context.run_config.get("batch-size", 128))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    # Support learning rate passed in message config or fall back to run_config
    cfg = msg.content.get("config")
    if cfg is not None and "lr" in cfg:
        learning_rate = float(cfg["lr"])
    else:
        learning_rate = float(context.run_config.get("learning-rate", 0.001))
    # 3. Load this client's partitioned training data
    trainloader, _ = load_client_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )
    class_weights = get_class_weights(device=str(device))
    # 4. Train local model on CPU
    avg_loss, accuracy, num_examples = train(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=learning_rate,
        device=str(device),
        class_weights=class_weights,
    )
    # 5. Package updated model parameters and metrics into reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics_record = MetricRecord(
        {
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "num-examples": int(num_examples),
        }
    )
    return Message(
        content=RecordDict({"arrays": model_record, "metrics": metrics_record}),
        reply_to=msg,
    )
@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate global parameters on the client's local validation partition."""
    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(context.run_config.get("device", "cpu"))
    partition_id = int(context.node_config.get("partition-id", 0))
    num_partitions = int(
        context.node_config.get(
            "num-partitions", context.run_config.get("num-clients", 2)
        )
    )
    batch_size = int(context.run_config.get("batch-size", 128))
    _, valloader = load_client_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )
    class_weights = get_class_weights(device=str(device))
    val_loss, val_acc, num_examples = test(
        model=model,
        testloader=valloader,
        device=str(device),
        class_weights=class_weights,
    )
    metrics_record = MetricRecord(
        {
            "eval_loss": float(val_loss),
            "eval_accuracy": float(val_acc),
            "num-examples": int(num_examples),
        }
    )
    return Message(
        content=RecordDict({"metrics": metrics_record}),
        reply_to=msg,
    )
