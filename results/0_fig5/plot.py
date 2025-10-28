from typing import List, Optional, Sequence, Tuple, Union, Dict

import torch
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

from graphxai.datasets import (
    Benzene,
    FluorideCarbonyl,
    BAHouse,
    BAWheel,
    AlkaneCarbonyl,
    Mutagenicity,
)
from torch_geometric.data import Data
from graphxai.utils.explanation import Explanation

# Utilities for conversions
from graphxai.utils.nx_conversion import to_networkx_conv

# Make project root importable first so that `from utils import ...` resolves to root `utils/`,
# not `data/utils.py` when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch
import os
from data.utils import get_dataset
from gnn.train_gnn_for_dataset import get_model
from utils import set_seed
from data.utils import filter_correct_data
from explainer.coa_explainer import CoAExplainer
from explainer.train_coaexplainer import train_coaexp
from graphxai.metrics.metrics_graph import graph_exp_acc_graph, graph_exp_faith_graph
import torch
from tqdm import tqdm
from graphxai.utils import Explanation
from graphxai.datasets import Benzene
from graphxai.explainers import SubgraphX
from results.utils import get_exp_method
from graphxai.utils.performance.load_exp import exp_exists_graph
import numpy as np
import argparse
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from metrics.edge_exp_metrics import (
    soft_mask_to_hard,
    fid_neg,
    fid_pos,
    jac_edge_max,
    jac_edge_all,
    faith_edge,
    sparsity_edge,
)

from data.Visualization import (
    visualize_graph_explanation,
    visualize_graph_explanation_style2,
)


dataset_name = "house_cycle"  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle
model_type = "GIN_3layer"  # GIN_3layer, GCN_3layer, GAT_2layer
exp_name = "rcex_top_0.22"  # gnnex, pgex_top_0.555, subx, rcex_top_0.555, coaex


sparsity = {
    "Benzene": 0.56,
    "FluorideCarbonyl": 0.66,
    "AlkaneCarbonyl": 0.82,
    "Mutagenicity": 0.45,
    "house_triangle": 0.63,
    "triangle_grid": 0.84,
    "house_cycle": 0.79,
}

nodesize_dict = {
    "Benzene": 900,
    "FluorideCarbonyl": 900,
    "AlkaneCarbonyl": 900,
    "Mutagenicity": 1200,
    "house_triangle": 600,
    "triangle_grid": 600,
    "house_cycle": 600,
}

if exp_name == "coaex":
    topk_ratio = None
else:
    topk_ratio = 1 - sparsity[dataset_name]


device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

set_seed(seed)

dataset = get_dataset(dataset_name, seed, device)

model = get_model(dataset, model_type, load_state=True)
model.eval()

correcr_train_index, correcr_val_index, correcr_test_index = filter_correct_data(
    dataset, model
)

exp_train_loader, _ = dataset.get_loader(index=correcr_train_index, batch_size=1)
exp_val_loader, evl_exp_ls = dataset.get_loader(index=correcr_val_index, batch_size=1)
exp_test_loader, test_exp_ls = dataset.get_loader(
    index=correcr_test_index, batch_size=1
)
add_args = {"batch": torch.zeros((1,)).long().to(device)}

exp_loc = os.path.join(
    "results",
    "1_explanation_accuracy",
    f"{model_type}",
    f"{dataset_name}",
    "metrics_result",
    "EXPS",
    f"{exp_name}",
)


id_path = os.path.join(
    os.path.dirname(__file__), f"sorted_indices{model_type}{dataset_name}.txt"
)
with open(id_path, "r", encoding="utf-8") as f:
    example_idx = [int(line.strip()) for line in f if line.strip()]

# data_idx = [i for i in example_idx[:3]]
# data_idx = [212, 1013, 1231]  # mutagenicity
# data_idx = [7007, 7211]  # benzene
# data_idx = [8139]  # fluoridecarbonyl
# data_idx = [439, 866]  # triangle_grid
# data_idx = [1001, 969]  # AlkaneCarbonyl
data_idx = [802]  # house_cycle
# data_idx = [501]  # house_triangle
for i in data_idx:
    data, gt_exp = dataset[i]
    data = data.to(device)
    out, _ = model(data.x, data.edge_index, **add_args)
    pred_class = out.argmax(dim=1).item()
    # explainer, forward_kwargs = get_exp_method(exp_name, model, torch.nn.CrossEntropyLoss().to(device), pred_class, data, device)
    # Try to load from existing dir:
    exp = exp_exists_graph(i, path=exp_loc, get_exp=True)
    if data.y.item() in [1]:  # evaluate_class
        fig, axes = plt.subplots(2, 1, figsize=(8, 16))
        axes = axes.flatten()
        print(f"exp.edge_imp: {exp.edge_imp}")
        visualize_graph_explanation_style2(
            data,
            exp,
            dataset_name=dataset_name,
            same_color_for_all_exps=False,
            draw_both_directions=False,
            show_all_edge_imp_values=False,
            top_ratio=topk_ratio,
            node_size=nodesize_dict[dataset_name],
            ax=axes[0],
            # ax=axes,
            # show=False,
        )
        axes[0].set_title(f"", fontsize=16)
        for e in gt_exp:
            print(f"gt_exp[i].edge_imp: {e.edge_imp}")
        visualize_graph_explanation_style2(
            data,
            gt_exp,
            dataset_name=dataset_name,
            same_color_for_all_exps=True,
            draw_both_directions=False,
            show_all_edge_imp_values=False,
            node_size=nodesize_dict[dataset_name],
            # top_k = 4,
            ax=axes[1],
            # show=False,
        )
        axes[1].set_title(f"Ground Truth", fontsize=16)
        plt.tight_layout()
        os.makedirs(
            os.path.join(os.path.dirname(__file__), f"{dataset_name}"), exist_ok=True
        )
        save_path = os.path.join(
            os.path.dirname(__file__),
            f"{dataset_name}",
            f"{dataset_name}_{i}_{exp_name}.svg",
        )
        plt.savefig(save_path, format="svg", dpi=300)
        # plt.show()
        # print("==========================")
