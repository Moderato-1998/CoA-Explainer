import torch
import os
from data.utils import get_dataset
from gnn.train_gnn_for_dataset import get_model
from utils import set_seed
from data.utils import filter_correct_data
import torch
from tqdm import tqdm
from results.utils import get_exp_method
from graphxai.utils.performance.load_exp import exp_exists_graph
import numpy as np
import argparse
from metrics.edge_exp_metrics import (
    soft_mask_to_hard,
    fid_neg,
    fid_pos,
    jac_edge_max,
    jac_edge_all,
    faith_edge,
    sparsity_edge,
)
import time
from graphxai.explainers import SubgraphX
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# parser = argparse.ArgumentParser(description='Run experiment with specified dataset, model, and explainer.')
# parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
# parser.add_argument('--model_type', type=str, required=True, help='Type of the model')
# parser.add_argument('--exp_name', type=str, required=True, help='Name of the explanation method')

# args = parser.parse_args()

# dataset_name = args.dataset_name
# model_type = args.model_type
# exp_name = args.exp_name

dataset_name = "Mutagenicity"  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle
model_type = "GIN_3layer"  # GCN_3layer, GIN_3layer, GAT_2layer
exp_name = "gnnex"  # gnnex subx

# 为了公平，需要在同一稀疏度下比较
if model_type in ["GIN_3layer"]:
    sparsity = {
        "Benzene": 0.56,
        "FluorideCarbonyl": 0.66,
        "AlkaneCarbonyl": 0.57,
        "Mutagenicity": 0.45,
        "house_triangle": 0.63,
        "triangle_grid": 0.84,
        "house_cycle": 0.79,
    }
elif model_type in ["GCN_3layer"]:
    sparsity = {
        "Benzene": 0.56,
    }
elif model_type in ["GAT_2layer"]:
    sparsity = {
        "Benzene": 0.69,
    }
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


evaluate_class = [1, 0]
ground_truth_1class = ["Benzene", "FluorideCarbonyl", "AlkaneCarbonyl", "Mutagenicity"]
if dataset_name in ground_truth_1class:
    evaluate_class = [1]

edge_gea = []
edge_gef = []
pos_fidelity = []
neg_fidelity = []
sparsity = []
t = []
for idx in tqdm(correcr_test_index):

    data, gt_exp = dataset[idx]
    data = data.to(device)

    out, _ = model(data.x, data.edge_index, **add_args)
    pred_class = out.argmax(dim=1).item()

    if data.y.item() in evaluate_class:

        max_nodes = int(data.num_nodes * topk_ratio)
        # explainer, forward_kwargs = get_exp_method(exp_name, model, torch.nn.CrossEntropyLoss().to(device), pred_class, data, max_nodes, device)

        explainer = SubgraphX(
            model,
            min_atoms=5,
            high2low=True,
            expand_atoms=20,
            subgraph_building_method="split",
        )
        forward_kwargs = {
            "x": data.x.to(device),
            "edge_index": data.edge_index.to(device),
            "label": pred_class,
            "max_nodes": max_nodes,
        }
        forward_kwargs["forward_kwargs"] = {
            "batch": torch.tensor([0]).long().to(device)
        }
        # Try to load from existing dir:
        exp = exp_exists_graph(idx, path=exp_loc, get_exp=True)
        if exp is None:
            t0 = time.time()
            exp = explainer.get_explanation_graph(**forward_kwargs)
            t1 = time.time()
            t.append(t1 - t0)
            os.makedirs(exp_loc, exist_ok=True)
            torch.save(
                exp, open(os.path.join(exp_loc, "exp_{:0>5d}.pt".format(idx)), "wb")
            )

        if exp_name in ["gnnex", "pgex"]:
            exp.edge_imp = soft_mask_to_hard(
                exp.edge_imp, type="ratio", value=topk_ratio
            )
        from graphxai.metrics.metrics_graph import (
            graph_exp_acc_graph,
            graph_exp_faith_graph,
        )

        neg_fidelity.append(fid_neg(data, exp, model, pred_class))

        pos_fidelity.append(fid_pos(data, exp, model, pred_class))

        edge_faith = faith_edge(exp, data, model, forward_kwargs=add_args)
        edge_gef.append(edge_faith)

        edge_acc = jac_edge_max(gt_exp, exp)
        edge_gea.append(edge_acc)

        # edge_faith = graph_exp_faith_graph(exp, data, model, forward_kwargs=add_args)
        # edge_gef.append(edge_faith)

        # edge_acc = graph_exp_acc_graph(gt_exp, exp)
        # edge_gea.append(edge_acc)

        sparsity.append(sparsity_edge(exp))


np.save(
    open(
        os.path.join(
            "results",
            "1_explanation_accuracy",
            f"{model_type}",
            f"{dataset_name}",
            "metrics_result",
            f"{exp_name}_gea.npy",
        ),
        "wb",
    ),
    np.array(edge_gea),
)
np.save(
    open(
        os.path.join(
            "results",
            "1_explanation_accuracy",
            f"{model_type}",
            f"{dataset_name}",
            "metrics_result",
            f"{exp_name}_gef.npy",
        ),
        "wb",
    ),
    np.array(edge_gef),
)
np.save(
    open(
        os.path.join(
            "results",
            "1_explanation_accuracy",
            f"{model_type}",
            f"{dataset_name}",
            "metrics_result",
            f"{exp_name}_fid+.npy",
        ),
        "wb",
    ),
    np.array(pos_fidelity),
)
np.save(
    open(
        os.path.join(
            "results",
            "1_explanation_accuracy",
            f"{model_type}",
            f"{dataset_name}",
            "metrics_result",
            f"{exp_name}_fid-.npy",
        ),
        "wb",
    ),
    np.array(neg_fidelity),
)
np.save(
    open(
        os.path.join(
            "results",
            "1_explanation_accuracy",
            f"{model_type}",
            f"{dataset_name}",
            "metrics_result",
            f"{exp_name}_sparsity.npy",
        ),
        "wb",
    ),
    np.array(sparsity),
)
np.save(
    open(
        os.path.join(
            "results",
            "1_explanation_accuracy",
            f"{model_type}",
            f"{dataset_name}",
            "metrics_result",
            f"{exp_name}_time.npy",
        ),
        "wb",
    ),
    np.array(t),
)
print(f"Average GEA: {sum(edge_gea) / len(edge_gea)}")
print(f"Average GEF: {sum(edge_gef) / len(edge_gef)}")
print(f"Average fid+: {sum(pos_fidelity) / len(pos_fidelity)}")
print(f"Average fid-: {sum(neg_fidelity) / len(neg_fidelity)}")
print(f"Average sparsity: {sum(sparsity) / len(sparsity)}")
print(f"Average time pre graph: {sum(t) / len(t)}")
