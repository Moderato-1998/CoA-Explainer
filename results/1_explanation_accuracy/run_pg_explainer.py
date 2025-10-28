import torch
import time
import os
from data.utils import get_dataset
from gnn.train_gnn_for_dataset import get_model
from utils import set_seed
from data.utils import filter_correct_data
import torch
from tqdm import tqdm
from graphxai.explainers import PGExplainer
from graphxai.utils.performance.load_exp import exp_exists_graph
import numpy as np
import argparse
from metrics.edge_exp_metrics import soft_mask_to_hard, fid_neg, fid_pos, jac_edge_max, jac_edge_all, faith_edge, sparsity_edge
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
exp_name = 'pgex'  # pgex
topk_ratio = 0.22

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

set_seed(seed)

dataset = get_dataset(dataset_name, seed, device)

model = get_model(dataset, model_type, load_state=True)
model.eval()

correcr_train_index, correcr_val_index, correcr_test_index = filter_correct_data(dataset, model)

# exp_train_loader, _ = dataset.get_loader(index=correcr_train_index, batch_size=1)
# exp_val_loader, evl_exp_ls = dataset.get_loader(index=correcr_val_index, batch_size=1)
# exp_test_loader, test_exp_ls = dataset.get_loader(index=correcr_test_index, batch_size=1)
add_args = {'batch': torch.zeros((1,)).long().to(device)}

exp_train_graphs = [dataset[i][0] for i in correcr_train_index]

params = {  
    'coeff_size': 1e-5,
    'coeff_ent': 3e-4,
    'eps': 1e-16,
    'max_epochs': 20,
    'lr': 0.01,
}

if model_type in ['GCN_3layer', 'GIN_3layer']:
    params['emb_layer_name'] = 'conv3'
elif model_type in ['GAT_2layer']:
    params['emb_layer_name'] = 'conv2'


explainer = PGExplainer(model=model,
                           emb_layer_name=params['emb_layer_name'],
                           coeff_size=params['coeff_size'],
                           coeff_ent=params['coeff_ent'],
                           eps=params['eps'],
                           max_epochs=params['max_epochs'],
                           lr=params['lr'],
                           explain_graph=True)


exp_parms_str = f"cs={params['coeff_size']}_ce={params['coeff_ent']}_eps={params['eps']}_lr={params['lr']}"
exp_model_path = os.path.join('results', '1_explanation_accuracy', f'{model_type}',
                                f'{dataset_name}',
                                'model_weights',
                                f'{exp_name}_{exp_parms_str}.pth')

if os.path.exists(exp_model_path):
    explainer.elayers = torch.load(exp_model_path)
    print(f"Load explainer: {exp_model_path}")
else:
    explainer.train_explanation_model(exp_train_graphs, forward_kwargs=add_args)
    torch.save(explainer.elayers, exp_model_path)

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

        # Try to load from existing dir:
        exp = exp_exists_graph(idx, path=exp_loc, get_exp=True)
        if exp is None:
            t0 = time.time()
            exp = explainer.get_explanation_graph(x=data.x.to(device),
                                             edge_index=data.edge_index.to(device),
                                             label=data.y.to(device),
                                             forward_kwargs=add_args,
                                             )
            t1 = time.time()
            t.append(t1 - t0)
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