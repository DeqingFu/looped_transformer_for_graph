import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import json

from model import LoopedTransformer
from data import ErdosRenyiGraphDataset

# Set up logging with a placeholder handler that will be replaced in the train function
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Just console logging initially
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphConnectivityDataset(Dataset):
    """PyTorch dataset for graph connectivity prediction."""

    def __init__(self, adj_matrices, conn_matrices):
        """
        Initialize dataset with precomputed adjacency and connectivity matrices.

        Args:
            adj_matrices: List of adjacency matrices
            conn_matrices: List of connectivity matrices
        """
        self.adj_matrices = adj_matrices
        self.conn_matrices = conn_matrices

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, idx):
        return self.adj_matrices[idx], self.conn_matrices[idx]


def precompute_dataset(dataset, num_samples):
    """
    Precompute all samples from a dataset.

    Args:
        dataset: ErdosRenyiGraphDataset
        num_samples: Number of samples to generate

    Returns:
        adj_matrices: List of adjacency matrices
        conn_matrices: List of connectivity matrices
    """
    adj_matrices = []
    conn_matrices = []

    for i in tqdm(range(num_samples), desc="Generating dataset"):
        adj_matrix, conn_matrix = dataset[i]
        adj_matrices.append(adj_matrix)
        conn_matrices.append(conn_matrix)

    return adj_matrices, conn_matrices


def binary_cross_entropy_loss(predictions, targets, reduction="mean"):
    """
    Custom BCE loss for adjacency matrix prediction.

    Args:
        predictions: Predicted adjacency matrix
        targets: Target adjacency matrix
        reduction: Reduction method (mean or sum)

    Returns:
        BCE loss
    """
    epsilon = 1e-7
    predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
    loss = -targets * torch.log(predictions) - (1 - targets) * torch.log(
        1 - predictions
    )

    if reduction == "mean":
        return torch.mean(loss)
    else:
        return torch.sum(loss)


def evaluate(model, dataloader, device, threshold=0.5):
    """
    Evaluate the model on a dataloader.

    Args:
        model: LoopedTransformer model
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        threshold: Threshold for binary prediction

    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0
    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    sample_count = 0

    with torch.no_grad():
        for adjacency_matrix, connectivity_matrix in dataloader:
            adjacency_matrix = adjacency_matrix.to(device).float()
            connectivity_matrix = connectivity_matrix.to(device).float()

            outputs, _ = model(adjacency_matrix)

            # Apply sigmoid to get probabilities
            outputs_sigmoid = torch.sigmoid(outputs)
            loss = binary_cross_entropy_loss(outputs_sigmoid, connectivity_matrix)

            # Convert to binary predictions
            pred_binary = (outputs_sigmoid > threshold).float()

            # Calculate metrics
            correct = (pred_binary == connectivity_matrix).float().mean()

            # Calculate precision, recall, F1 (for positive class)
            true_positives = (pred_binary * connectivity_matrix).sum()
            predicted_positives = pred_binary.sum()
            actual_positives = connectivity_matrix.sum()

            precision = true_positives / (predicted_positives + 1e-7)
            recall = true_positives / (actual_positives + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

            total_loss += loss.item() * adjacency_matrix.size(0)
            accuracy_sum += correct.item() * adjacency_matrix.size(0)
            precision_sum += precision.item() * adjacency_matrix.size(0)
            recall_sum += recall.item() * adjacency_matrix.size(0)
            f1_sum += f1.item() * adjacency_matrix.size(0)
            sample_count += adjacency_matrix.size(0)

    metrics = {
        "loss": total_loss / sample_count,
        "accuracy": accuracy_sum / sample_count,
        "precision": precision_sum / sample_count,
        "recall": recall_sum / sample_count,
        "f1": f1_sum / sample_count,
    }

    return metrics


def train(args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments to config.json
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        # Convert args to a dictionary and save as JSON
        args_dict = vars(args)
        # Convert non-serializable objects to strings
        for k, v in args_dict.items():
            if not isinstance(
                v, (int, float, str, bool, list, dict, tuple, type(None))
            ):
                args_dict[k] = str(v)
        # Convert tuples to lists for JSON serialization
        for k, v in args_dict.items():
            if isinstance(v, tuple):
                args_dict[k] = list(v)
        json.dump(args_dict, f, indent=4)

    logger.info(f"Configuration saved to: {config_path}")

    # Configure logging to write to the output directory
    log_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()

    # Remove any existing file handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    # Add the new file handler
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Initializing datasets...")
    train_dataset = ErdosRenyiGraphDataset(
        num_samples=args.train_samples,
        num_nodes=args.graph_size,
        p=args.edge_probability,
        sample_p=args.sample_p,
        p_range=args.p_range,
        add_self_loops=args.add_self_loops,
    )

    val_raw_dataset = ErdosRenyiGraphDataset(
        num_samples=args.val_samples,
        num_nodes=args.graph_size,
        p=args.edge_probability,
        sample_p=args.sample_p,
        p_range=args.p_range,
        add_self_loops=args.add_self_loops,
    )

    # Precompute all datasets
    logger.info("Precomputing datasets for evaluation ...")

    val_adj_matrices, val_conn_matrices = precompute_dataset(
        val_raw_dataset, args.val_samples
    )

    # Compute average label value to understand class distribution
    avg_label = torch.stack(val_conn_matrices).float().mean().item()
    logger.info(
        f"Average connectivity in training set: {avg_label:.4f} (class balance)"
    )

    # Create PyTorch datasets
    val_dataset = GraphConnectivityDataset(val_adj_matrices, val_conn_matrices)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    logger.info("Initializing model...")
    model = LoopedTransformer(
        graph_size=args.graph_size,
        n_loop=args.n_loop,
        hidden_size=args.hidden_size,
        read_in_method=args.read_in_method,
        num_attention_heads=args.num_attention_heads,
    ).to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    # Initialize learning rate scheduler with cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.min_learning_rate
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        train_samples = 0

        # Store current learning rate for plotting
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)
        logger.info(f"Current learning rate: {current_lr:.6f}")

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for adjacency_matrix, connectivity_matrix in progress_bar:
            adjacency_matrix = adjacency_matrix.to(device).float()
            connectivity_matrix = connectivity_matrix.to(device).float()

            optimizer.zero_grad()

            outputs, layer_outputs = model(adjacency_matrix)

            # Apply sigmoid to get probabilities
            outputs_sigmoid = torch.sigmoid(outputs)

            # Compute loss
            loss = binary_cross_entropy_loss(outputs_sigmoid, connectivity_matrix)

            # Add auxiliary losses from intermediate layers
            if args.use_auxiliary_loss and len(layer_outputs) > 0:
                aux_loss = 0
                for i, layer_output in enumerate(layer_outputs):
                    layer_output_sigmoid = torch.sigmoid(layer_output)
                    aux_loss += binary_cross_entropy_loss(
                        layer_output_sigmoid, connectivity_matrix
                    ) * (0.5 ** (len(layer_outputs) - i))
                loss += args.aux_loss_weight * aux_loss

            loss.backward()

            # Gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

            # Calculate training accuracy
            pred_binary = (outputs_sigmoid > 0.5).float()
            train_accuracy = (pred_binary == connectivity_matrix).float().mean().item()

            batch_size = adjacency_matrix.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_accuracy += train_accuracy * batch_size
            train_samples += batch_size

            progress_bar.set_postfix({"loss": loss.item(), "acc": train_accuracy})

        avg_train_loss = total_train_loss / train_samples
        avg_train_accuracy = total_train_accuracy / train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation phase
        val_metrics = evaluate(model, val_dataloader, device)
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])

        # Update learning rate with cosine scheduler (per epoch)
        scheduler.step()

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Accuracy: {avg_train_accuracy:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)

            # Save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )

            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.title("Learning Rate Schedule")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LoopedTransformer for graph connectivity prediction"
    )

    # Data parameters
    parser.add_argument(
        "--graph_size", type=int, default=32, help="Number of nodes in each graph"
    )
    parser.add_argument(
        "--edge_probability",
        type=float,
        default=0.1,
        help="Probability of edge creation",
    )
    parser.add_argument(
        "--sample_p", type=bool, default=True, help="Whether to sample p from a range"
    )
    parser.add_argument(
        "--p_range",
        type=float,
        nargs=2,
        default=(0.02, 0.2),
        help="Range for sampling p if sample_p is True",
    )
    parser.add_argument(
        "--add_self_loops", type=bool, default=True, help="Whether to add self-loops"
    )
    parser.add_argument(
        "--train_samples", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--val_samples", type=int, default=200, help="Number of validation samples"
    )

    # Model parameters
    parser.add_argument(
        "--n_loop", type=int, default=3, help="Number of transformer loops"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden size of the transformer"
    )
    parser.add_argument(
        "--read_in_method",
        type=str,
        default="linear",
        choices=["linear", "zero_pad"],
        help="Method to read in adjacency matrix",
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=4, help="Number of attention heads"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine decay",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--use_auxiliary_loss",
        type=bool,
        default=True,
        help="Whether to use auxiliary loss",
    )
    parser.add_argument(
        "--aux_loss_weight", type=float, default=0.5, help="Weight for auxiliary loss"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for saved models",
    )

    args = parser.parse_args()

    train(args)
