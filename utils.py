import torch 
from torch.utils.data import Dataset
import numpy as np 
import random
from model import LoopedTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, Tuple
from tqdm import tqdm
import matplotlib
from matplotlib.colors import LogNorm, SymLogNorm, FuncNorm
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.weight"] = 'normal'
plt.rcParams["axes.labelweight"] = "normal"

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_qk_matrix(model: LoopedTransformer):
    """
    Extract the QK matrix from the LoopedTransformer model.

    Args:
        model: LoopedTransformer model

    Returns:
        QK matrix
    """
    q_matrix = model.layer.attention.self.query.weight.data
    k_matrix = model.layer.attention.self.key.weight.data
    qk_matrix = torch.matmul(q_matrix, k_matrix.T)
    return qk_matrix


def plot_qk_matrix(qk_matrix: torch.Tensor, epoch: int, output_dir: str):
    """
    Plot the QK matrix using seaborn for better visualization.

    Args:
        qk_matrix: The QK matrix to plot
        epoch: Current epoch number
        output_dir: Directory to save the plot
    """
    # Create directory for QK visualizations if it doesn't exist
    qk_vis_dir = os.path.join(output_dir, "QK_vis")
    os.makedirs(qk_vis_dir, exist_ok=True)

    # Convert tensor to numpy array for plotting
    qk_numpy = qk_matrix.cpu().numpy()

    # Create figure with appropriate size
    plt.figure(figsize=(10, 8))

    # Create heatmap with seaborn
    ax = sns.heatmap(
        qk_numpy,
        cmap="viridis",
        vmin=qk_numpy.min(),
        vmax=qk_numpy.max(),
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    # Add title and axis labels
    plt.title(f"QK Matrix - Epoch {epoch}")
    plt.xlabel("Key dimension")
    plt.ylabel("Query dimension")

    # Save the figure
    plt.tight_layout()
    filename = f"qk_matrix_epoch{epoch:03d}.png"
    plt.savefig(os.path.join(qk_vis_dir, filename))
    plt.close()


def plot_qk_matrix_multi_head(model: LoopedTransformer, epoch: int, output_dir: str):
    """
    Extract and plot QK attention matrices for each attention head.

    Args:
        model: The LoopedTransformer model
        epoch: Current epoch number
        output_dir: Directory to save the plots
    """
    # Create directory for QK visualizations if it doesn't exist
    qk_vis_dir = os.path.join(output_dir, "QK_vis")
    os.makedirs(qk_vis_dir, exist_ok=True)

    # Extract QK matrix
    qk_matrix = extract_qk_matrix(model)

    # Get model parameters
    hidden_size = model.hidden_size
    num_heads = model.config.num_attention_heads
    head_size = hidden_size // num_heads

    # Plot overall QK matrix
    plot_qk_matrix(qk_matrix, epoch, output_dir)

    # If we have multiple attention heads, visualize each head separately
    if num_heads > 1:
        # Create a figure for all heads
        fig, axes = plt.subplots(
            1, num_heads, figsize=(num_heads * 5, 5), squeeze=False
        )

        # For each attention head
        for head_idx in range(num_heads):
            # Calculate start and end indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size

            # Extract QK matrix for this head
            head_qk = qk_matrix[start_idx:end_idx, start_idx:end_idx]
            head_qk_numpy = head_qk.cpu().numpy()

            # Plot on corresponding subplot
            ax = axes[0, head_idx]
            sns.heatmap(
                head_qk_numpy,
                cmap="viridis",
                vmin=head_qk_numpy.min(),
                vmax=head_qk_numpy.max(),
                square=True,
                ax=ax,
                cbar=True if head_idx == num_heads - 1 else False,
            )
            ax.set_title(f"Head {head_idx+1}")

            # Only show y labels for the first plot
            if head_idx > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

        # Add overall title
        fig.suptitle(f"QK Matrices by Attention Head - Epoch {epoch}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title

        # Save the multi-head figure
        filename = f"qk_matrix_heads_epoch{epoch:03d}.png"
        plt.savefig(os.path.join(qk_vis_dir, filename))
        plt.close()


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

def compute_f1(pred_binary, connectivity_matrix):
    true_positives = (pred_binary * connectivity_matrix).sum()
    predicted_positives = pred_binary.sum()
    actual_positives = connectivity_matrix.sum()

    precision = true_positives / (predicted_positives + 1e-7)
    recall = true_positives / (actual_positives + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1, precision, recall

def transform(x):
    return - np.log ( 1 - x)

def inverse_transform(y):
    return 1 - np.exp(-y)

def plot_heatmap(heatmap, xticklabels, yticklabels, \
                 metric="accuracy", output_dir="plots"):
    fig, ax = plt.subplots(figsize=(16,12), dpi=400)

    annotation = np.empty_like(heatmap).astype(str); annotation[:] = ''
    full_annotation=True
    boxes = []
    for col, _ in enumerate(heatmap):
        row = np.argmax(heatmap[col])
        boxes.append((col, row))
        annotation[col, row] = f"{heatmap[col, row]:.3f}".lstrip('0')
        if row - 1 >= 0:
            annotation[col, row-1] = f"{heatmap[col, row-1]:.3f}".lstrip('0')
        if row + 1 < heatmap.shape[1]:
            annotation[col, row+1] = f"{heatmap[col, row+1]:.3f}".lstrip('0')
    if full_annotation:
        for col, _ in enumerate(heatmap):
            for row, _ in enumerate(heatmap[col]):
                annotation[col, row] = f"{heatmap[col, row]:.3f}".lstrip('0')
    s = sns.heatmap(heatmap.T, cmap='PuRd', 
                    ax=ax, annot=annotation.T, norm=FuncNorm((transform, inverse_transform), vmin=0.5, vmax=0.99),
                    xticklabels=xticklabels, yticklabels=yticklabels, 
                    fmt='s', vmin=0.5, vmax=0.99, cbar=False,
                    cbar_kws={"orientation": "horizontal", "pad":0.02},
                    annot_kws={"fontsize": 18},
                    linewidths=0.0, rasterized=True)
    for box in boxes:
        ax.add_patch(Rectangle(box, 1, 1, fill=False, edgecolor='yellow', lw=8))

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='y', rotation=0)

    plt.title(f"Average {metric.capitalize()} Matrix")
    plt.xlabel("Number of Loops")
    plt.ylabel(r"Matrix Power $A^k$")
    plt.savefig(output_dir + f"/{metric}_matrix.png", dpi=400, bbox_inches="tight", pad_inches=0.05)