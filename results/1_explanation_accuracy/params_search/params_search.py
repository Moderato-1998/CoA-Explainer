import os
import time
import itertools
import numpy as np
import torch
from tqdm import tqdm

from utils import set_seed
from data.utils import get_dataset, filter_correct_data
from gnn.train_gnn_for_dataset import get_model
from explainer.coa_explainer import CoAExplainer
from explainer.train_coaexplainer import train_coaexp, evaluate_model


def _infer_node_emb_dim(mdl, device, dataset):
    # Try to read from final linear layer in_features
    if hasattr(mdl, 'lin') and hasattr(mdl.lin, 'in_features'):
        try:
            return int(mdl.lin.in_features)
        except Exception:
            pass
    # Fallback: forward one graph to get n_emb width
    try:
        sample_idx = dataset.train_index[0] if len(dataset.train_index) > 0 else 0
        g, _ = dataset[sample_idx]
        g = g.to(device)
        b = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        mdl.eval()
        with torch.no_grad():
            _, n_emb = mdl(g.x, g.edge_index, b)
        return int(n_emb.size(1))
    except Exception:
        return 64


def main():
    # Basic config
    dataset_name = 'Benzene'  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, grid_triangle, house_cycle
    model_type = 'GIN_3layer'          # 'GIN_3layer', 'GCN_3layer', 'GAT_2layer'
    exp_name = 'coaex'               # 'coaex', 'coaex_base'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    set_seed(seed)

    # Prepare dataset/model
    dataset = get_dataset(dataset_name, seed, device)
    model = get_model(dataset, model_type, load_state=True)
    model.eval()

    # Correctly predicted indices and loaders
    correcr_train_index, correcr_val_index, correcr_test_index = filter_correct_data(dataset, model)
    train_loader, _ = dataset.get_loader(index=correcr_train_index[:100], batch_size=1)
    val_loader, evl_exp_ls = dataset.get_loader(index=correcr_val_index[:100], batch_size=1)

    # Search space (grid)
    param_grid = {
        'actor_lr': [1e-4, 5e-4, 1e-3],
        'critic_lr': [5e-5, 1e-4, 3e-4],
        'batch_size': [64, 128],
        'buffer_capacity': [1500, 2000],
        'exploration_noise_std': [0.02, 0.05, 0.1],
        'num_episodes_per_graph': [2,8,16],
        'num_epochs': [20],
    }

    # Output dirs/files (for model cache and logs)
    root_dir = os.path.join('results', '1_explanation_accuracy', model_type, dataset_name)
    model_dir = os.path.join(root_dir, 'model_weights')
    report_dir = os.path.join(root_dir, 'params_search')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    report_log = os.path.join(report_dir, f'{exp_name}_grid_results.log')
    # Prepare log file helper
    def _log(line: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(report_log, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {line}\n")
    _log(f"Start grid search | dataset={dataset_name}, model={model_type}, exp={exp_name}")
    _log(f"LOG: {report_log}")

    # Iterate grid
    best = {
        'score': -1e9,
        'metrics': None,
        'params': None,
        'model_path': None,
    }

    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))
    print(f"Total combinations: {len(combos)}")

    node_emb_dim = _infer_node_emb_dim(model, device, dataset)
    print(f"Inferred node embedding dim: {node_emb_dim}")

    for combo in combos:
        cfg = dict(zip(keys, combo))
        tag = (
            f"alr={cfg['actor_lr']}_clr={cfg['critic_lr']}"
            f"_bs={cfg['batch_size']}_bc={cfg['buffer_capacity']}"
            f"_ens={cfg['exploration_noise_std']}_epg={cfg['num_episodes_per_graph']}"
            f"_ep={cfg['num_epochs']}"
        )
        model_path = os.path.join(model_dir, f"{exp_name}_{tag}.pth")
        print(f"\n=== HP Try: {tag} ===")
        _log(f"HP Try: {tag}")

        # Build explainer per config
        explainer = CoAExplainer(
            gnn_model=model,
            num_node_features=dataset.graphs[0].x.size(1),
            max_nodes=dataset.max_nodes,
            node_embedding_dim=node_emb_dim,
            actor_lr=cfg['actor_lr'],
            critic_lr=cfg['critic_lr'],
            device=device,
            batch_size=cfg['batch_size'],
            buffer_capacity=cfg['buffer_capacity'],
            exploration_noise_std=cfg['exploration_noise_std'],
        )

        # Train (or reuse) and evaluate on val set
        if os.path.exists(model_path):
            explainer.load_model(model_path)
            print(f"Reuse existing model: {model_path}")
        else:
            info = train_coaexp(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        evl_exp_ls=evl_exp_ls,
                        explainer=explainer,
                        num_epochs=cfg['num_epochs'],
                        num_episodes_per_graph=cfg['num_episodes_per_graph'],
                        save_path=model_path,
                    )
            # Write per-epoch training info to log
            for line in info:
                _log(line)
            # ensure best weights reloaded
            if os.path.exists(model_path):
                explainer.load_model(model_path)

        val_acc, val_sparsity, val_fid, val_exp_acc = evaluate_model(
            explainer, val_loader, evl_exp_ls, epoch=cfg['num_epochs']-1, num_epochs=cfg['num_epochs']
        )
        # combined = 0.35 * float(val_sparsity) + float(val_exp_acc)
        combined = float(val_exp_acc)

        # Log metrics for this parameter combo and add a blank line
        _log(
            "metrics: "
            + str({
            'val_acc': round(float(val_acc), 2),
            'val_fidelity': round(float(val_fid), 2),
            'val_sparsity': round(float(val_sparsity), 2),
            'val_exp_acc': round(float(val_exp_acc), 2),
            'combined': round(combined, 2),
            })
        )
        with open(report_log, 'a', encoding='utf-8') as f:
            f.write("\n")

        # Track best
        if combined > best['score']:
            best.update({
                'score': combined,
                'metrics': {
                    'val_acc': float(val_acc),
                    'val_fidelity': float(val_fid),
                    'val_sparsity': float(val_sparsity),
                    'val_exp_acc': float(val_exp_acc),
                    'combined': combined,
                },
                'params': cfg,
                'model_path': model_path,
            })
        print(f"Val: Acc {val_acc:.2f}%, Fid {val_fid:.3f}, Spar {val_sparsity:.3f}, ExpAcc {val_exp_acc:.2f} | Combined {combined:.3f}")

    # Summary
    print("\n===== Best Config =====")
    print(best['params'])
    print(best['metrics'])
    print(f"Model: {best['model_path']}")
    # Write Summary to log
    _log("===== Best Config =====")
    _log(f"params: {best['params']}")
    _log(f"metrics: {best['metrics']}")
    _log(f"model: {best['model_path']}")
    _log("Grid search finished.")


if __name__ == '__main__':
    main()