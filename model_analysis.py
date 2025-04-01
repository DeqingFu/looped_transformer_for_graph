from model import LoopedTransformer
import json 
import torch
from data import ErdosRenyiGraphDataset, ErdosRenyiTwoGraphsDataset
from utils import set_seed, precompute_dataset, compute_f1, plot_heatmap
from tqdm import tqdm
import pdb 
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = "outputs_32_nodes/20250331_231018"

config = json.load(open(ckpt_dir + "/config.json"))
model = LoopedTransformer(
    graph_size=config["graph_size"],
    n_loop=config["n_loop"],
    hidden_size=config["hidden_size"],
    read_in_method=config["read_in_method"],
    num_attention_heads=config["num_attention_heads"],
    tie_qk=config["tie_qk"],
    layernorm_type=config["layernorm_type"],
)

# Load the best model state dict
checkpoint_path = f"{ckpt_dir}/last_model.pt"
state_dict = torch.load(checkpoint_path, weights_only=True)['model_state_dict']
model.load_state_dict(state_dict)
model.to(device)  # Move the model to the appropriate device
# Set the model to evaluation mode
model.eval() 

test_n_samples = 1024
test_n_loop = config["n_loop"]
model.n_loop = test_n_loop

set_seed(config["seed"])  # Set the random seed for reproducibility
# Create a dataset instance
eval_dataset = ErdosRenyiGraphDataset(
    num_samples=test_n_samples,
    num_nodes=config["graph_size"],
    p = config["edge_probability"],
    sample_p=config["sample_p"],
    p_range=config["p_range"],
    add_self_loops=config["add_self_loops"],
)
# Precompute the dataset
adj_matrices, conn_matrices = precompute_dataset(dataset=eval_dataset, num_samples=test_n_samples)

accuracy_matrix = torch.zeros(test_n_loop+1, config['graph_size']+1)
f1_matrix = torch.zeros(test_n_loop+1, config['graph_size']+1)

xticklabels = [str(i) for i in range(test_n_loop+1)]
yticklabels = [str(i) for i in range(config['graph_size']+1)]

for adj_matrix in tqdm(adj_matrices):
    adj_matrix = adj_matrix.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        adj_matrix = adj_matrix.to(device)
        _, layer_outputs = model(adj_matrix)
    for i, layer_output in enumerate(layer_outputs):
        # Compute accuracy for each layer
        outputs_sigmoid = torch.sigmoid(layer_output)
        pred_matrix = (outputs_sigmoid > 0.5).float()
        for j in range(config['graph_size']+1):
            reachability_matrix = (torch.matrix_power(adj_matrix, j) > 0).float()
            # Compute accuracy
            accuracy = (reachability_matrix == pred_matrix).float().mean().item()
            accuracy_matrix[i, j] += accuracy
            # Compute F1 score
            f1 = compute_f1(pred_matrix, reachability_matrix)[0].float().item()
            f1_matrix[i, j] += f1
# Normalize the accuracy matrix
accuracy_matrix /= len(adj_matrices)
f1_matrix /= len(adj_matrices)


# Plot the matrices
plot_heatmap(accuracy_matrix, xticklabels=xticklabels, yticklabels=yticklabels, metric="accuracy", output_dir=ckpt_dir)
plot_heatmap(f1_matrix, xticklabels=xticklabels, yticklabels=yticklabels,  metric="f1", output_dir=ckpt_dir)