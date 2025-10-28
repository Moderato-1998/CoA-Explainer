import torch
import os
import time
from data.utils import get_dataset
from gnn.train_gnn_for_dataset import get_model
from utils import set_seed
from data.utils import filter_correct_data
from explainer.coa_explainer import CoAExplainer
from explainer.coa_explainer_base import CoAExplainer_Base
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

dataset_name = "AlkaneCarbonyl"  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle
model_type = "GIN_3layer"  # GIN_3layer, GCN_3layer, GAT_2layer
exp_name = "coaex"  # coaex, coaex_base

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

set_seed(seed)

dataset = get_dataset(dataset_name, seed, device)

model = get_model(dataset, model_type, load_state=True)
model.eval()

state_encode = get_model(dataset, "GIN_3layer", load_state=True)

correcr_train_index, correcr_val_index, correcr_test_index = filter_correct_data(
    dataset, model
)

exp_train_loader, _ = dataset.get_loader(index=correcr_train_index[:1000], batch_size=1)
exp_val_loader, evl_exp_ls = dataset.get_loader(index=correcr_val_index, batch_size=1)
exp_test_loader, test_exp_ls = dataset.get_loader(
    index=correcr_test_index, batch_size=1
)
add_args = {"batch": torch.zeros((1,)).long().to(device)}


params = {
    "actor_lr": 0.003,
    "critic_lr": 1e-05,
    "batch_size": 64,
    "buffer_capacity": 1000,
    "exploration_noise_std": 0.1,
    "num_episodes_per_graph": 1,
    "steps_per_episode": 2,
}
if dataset_name in ["Benzene", "FluorideCarbonyl"]:
    params["num_epochs"] = 20
else:
    params["num_epochs"] = 15

if exp_name == "coaex-base":
    explainer = CoAExplainer_Base(
        gnn_model=model,
        # state_encode=state_encode,
        num_node_features=dataset.graphs[0].x.size(1),
        max_nodes=dataset.max_nodes,
        # node_embedding_dim=state_encode.hidden_channels,
        node_embedding_dim=model.hidden_channels,
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        device=device,
        batch_size=params["batch_size"],
        buffer_capacity=params["buffer_capacity"],
        exploration_noise_std=params["exploration_noise_std"],
    )
else:
    explainer = CoAExplainer(
        gnn_model=model,
        state_encode=state_encode,
        num_node_features=dataset.graphs[0].x.size(1),
        max_nodes=dataset.max_nodes,
        node_embedding_dim=state_encode.hidden_channels,
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        device=device,
        batch_size=params["batch_size"],
        buffer_capacity=params["buffer_capacity"],
        exploration_noise_std=params["exploration_noise_std"],
        steps_per_episode=params["steps_per_episode"],
    )


exp_parms_str = f"alr={params['actor_lr']}_clr={params['critic_lr']}_bs={params['batch_size']}_bc={params['buffer_capacity']}_ens={params['exploration_noise_std']}"
exp_model_path = os.path.join(
    "results",
    "1_explanation_accuracy",
    f"{model_type}",
    f"{dataset_name}",
    "model_weights",
    f"{exp_name}_{exp_parms_str}.pth",
)

evaluate_class = [1, 0]
ground_truth_1class = ["Benzene", "FluorideCarbonyl", "AlkaneCarbonyl", "Mutagenicity"]
if dataset_name in ground_truth_1class:
    evaluate_class = [1]

if os.path.exists(exp_model_path):
    explainer.load_model(exp_model_path)
    print(f"Load explainer: {exp_model_path}")
else:
    train_coaexp(
        train_loader=exp_train_loader,
        val_loader=exp_val_loader,
        evl_exp_ls=evl_exp_ls,
        explainer=explainer,
        num_epochs=params["num_epochs"],
        num_episodes_per_graph=params["num_episodes_per_graph"],
        save_path=exp_model_path,
        evaluate_class=evaluate_class,
    )

explainer.actor.eval()
explainer.critic.eval()
explainer.state_gnn.eval()

exp_loc = os.path.join(
    "results",
    "1_explanation_accuracy",
    f"{model_type}",
    f"{dataset_name}",
    "metrics_result",
    "EXPS",
    f"{exp_name}",
)


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

        # Try to load from existing dir:
        exp = exp_exists_graph(idx, path=exp_loc, get_exp=True)
        if exp is None:
            t0 = time.time()
            _, _, _, _, _, _, _, _, edge_mask = explainer.evaluate_graph(data)
            t1 = time.time()
            t.append(t1 - t0)
            exp = Explanation(
                feature_imp=None,
                node_imp=None,
                edge_imp=edge_mask,
                graph=data,
            )
            os.makedirs(exp_loc, exist_ok=True)
            torch.save(
                exp, open(os.path.join(exp_loc, "exp_{:0>5d}.pt".format(idx)), "wb")
            )

        # evaluate
        neg_fidelity.append(fid_neg(data, exp, model, pred_class))
        pos_fidelity.append(fid_pos(data, exp, model, pred_class))

        edge_faith = faith_edge(exp, data, model, forward_kwargs=add_args)
        edge_gef.append(edge_faith)

        edge_acc = jac_edge_max(gt_exp, exp)
        edge_gea.append(edge_acc)

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
