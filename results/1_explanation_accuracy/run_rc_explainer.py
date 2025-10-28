import torch
import time
import copy
import os
from data.utils import get_dataset
from gnn.train_gnn_for_dataset import get_model
from utils import set_seed
from data.utils import filter_correct_data
from graphxai.metrics.metrics_graph import graph_exp_acc_graph, graph_exp_faith_graph
import torch
from tqdm import tqdm
from graphxai.utils import Explanation
from graphxai.datasets import Benzene
from graphxai.explainers import PGExplainer
from results.utils import get_exp_method
from graphxai.utils.performance.load_exp import exp_exists_graph
import numpy as np
import argparse
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from metrics.edge_exp_metrics import soft_mask_to_hard, fid_neg, fid_pos, jac_edge_max, jac_edge_all, faith_edge, sparsity_edge
from explainer.baseline.rc_explainer import RC_Explainer_Batch_star, train_policy
from explainer.baseline.rc_model_adapter import RCModelAdapter
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

dataset_name = 'house_cycle'  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle
model_type = 'GIN_3layer'  # GIN_3layer, GCN_3layer, GAT_2layer
exp_name = 'rcex'  # rcex
topk_ratio = 0.22

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

set_seed(seed)

dataset = get_dataset(dataset_name, seed, device)

model = get_model(dataset, model_type, load_state=True)
model.eval()

correcr_train_index, correcr_val_index, correcr_test_index = filter_correct_data(dataset, model)

exp_train_loader, _ = dataset.get_loader(index=correcr_train_index, batch_size=1)
exp_val_loader, evl_exp_ls = dataset.get_loader(index=correcr_val_index, batch_size=1)
exp_test_loader, test_exp_ls = dataset.get_loader(index=correcr_test_index, batch_size=1)
add_args = {'batch': torch.zeros((1,)).long().to(device)}

# exp_train_graphs = [dataset[i][0] for i in correcr_train_index]


_hidden_size = model.hidden_channels
_num_labels = 2
debias_flag = False
topN = None
batch_size = 64
scope = 0.1

model_1 = copy.deepcopy(model)
# Wrap base model with adapter to provide RC-required interfaces without changing original model
rc_ready_model = RCModelAdapter(model_1)

explainer = RC_Explainer_Batch_star(_model=rc_ready_model, _num_labels=_num_labels,
                                           _hidden_size=_hidden_size, _use_edge_attr=False).to(device)
lr = 0.0001
weight_decay = 0.00001
reward_mode = 'mutual_info'

optimizer = explainer.get_optimizer(lr=lr, weight_decay=weight_decay, scope=scope)

exp_parms_str = f"bs={batch_size}_lr={lr}_wd={weight_decay}_rm={reward_mode}"
exp_model_path = os.path.join('results', '1_explanation_accuracy', f'{model_type}',
                                f'{dataset_name}',
                                'model_weights',
                                f'{exp_name}_{exp_parms_str}.pth')

if os.path.exists(exp_model_path):
    explainer.load_policy_net(path=exp_model_path)
    print(f"Load explainer: {exp_model_path}")
else:
    rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec = \
        train_policy(explainer, rc_ready_model, exp_train_loader, exp_test_loader, optimizer, topk_ratio,
                     debias_flag=debias_flag, topN=topN, batch_size=batch_size, reward_mode=reward_mode,
                     save_model_path=exp_model_path)
    # train_policy auto save the best model

exp_loc = os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', 'EXPS', f'{exp_name}_top_{topk_ratio}')

evaluate_class = [1, 0]
ground_truth_1class = ['Benzene', 'FluorideCarbonyl', 'AlkaneCarbonyl', 'Mutagenicity']
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

        
        exp = exp_exists_graph(idx, path=exp_loc, get_exp=True)
        if exp is None:
            t0 = time.time()
            edge_mask = explainer.get_edge_mask(graph=data, topk_ratio=topk_ratio, mode='soft')
            t1 = time.time()
            t.append(t1 - t0)

            exp = Explanation(
                    feature_imp=None,
                    node_imp=None,
                    edge_imp=edge_mask,
                    graph=data,
                    )
            os.makedirs(exp_loc, exist_ok=True)
            torch.save(exp, open(os.path.join(exp_loc, 'exp_{:0>5d}.pt'.format(idx)), 'wb'))
        
        if exp_name in ['gnnex', 'pgex']:
            exp.edge_imp = soft_mask_to_hard(exp.edge_imp, type='ratio', value=topk_ratio)

        # evaluate
        neg_fidelity.append(fid_neg(data, exp, model, pred_class))
        pos_fidelity.append(fid_pos(data, exp, model, pred_class))

        edge_faith = faith_edge(exp, data, model, forward_kwargs=add_args)
        edge_gef.append(edge_faith)
        
        edge_acc = jac_edge_max(gt_exp, exp)
        edge_gea.append(edge_acc)

        sparsity.append(sparsity_edge(exp))

np.save(open(os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', f'{exp_name}_gea.npy'), 'wb'), np.array(edge_gea))
np.save(open(os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', f'{exp_name}_gef.npy'), 'wb'), np.array(edge_gef))
np.save(open(os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', f'{exp_name}_fid+.npy'), 'wb'), np.array(pos_fidelity))
np.save(open(os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', f'{exp_name}_fid-.npy'), 'wb'), np.array(neg_fidelity))
np.save(open(os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', f'{exp_name}_sparsity.npy'), 'wb'), np.array(sparsity))
np.save(open(os.path.join('results', '1_explanation_accuracy', f'{model_type}', f'{dataset_name}', 'metrics_result', f'{exp_name}_time.npy'), 'wb'), np.array(t))
print(f"Average GEA: {sum(edge_gea) / len(edge_gea)}")
print(f"Average GEF: {sum(edge_gef) / len(edge_gef)}")
print(f"Average fid+: {sum(pos_fidelity) / len(pos_fidelity)}")
print(f"Average fid-: {sum(neg_fidelity) / len(neg_fidelity)}")
print(f"Average sparsity: {sum(sparsity) / len(sparsity)}")
print(f"Average time pre graph: {sum(t) / len(t)}")