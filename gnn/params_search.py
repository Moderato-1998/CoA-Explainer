import os
import torch
import json
from graphxai.datasets import AlkaneCarbonyl, Mutagenicity, Benzene, FluorideCarbonyl
from graphxai.datasets import BAHouse, BADiamond, BAWheel, BACycle
from gnn.utils import train_model
from gnn.models import GCN_3layer, GIN_3layer, GAT_3layer
import random
import numpy as np
import torch_geometric
from data.utils import get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


hparams = {
    'GCN_3layer': {
        'Benzene': {    # Test F1: 0.8852, Test AUROC: 0.9451
            'hidden_channels': 64,
            'lr': 5e-3,
            'weight_decay': 0,
            'epochs': 1000
        },
        'Mutagenicity': {   # Test F1: 0.9588, Test AUROC: 0.9901
            'hidden_channels': 64,
            'lr': 1e-2,
            'weight_decay': 0,
            'epochs': 1000
        },
        'AlkaneCarbonyl': {  # Test F1: 0.9412, Test AUROC: 0.9756
            'hidden_channels': 64,
            'lr': 3e-3,
            'weight_decay': 0,
            'epochs': 1000
        },
        'FluorideCarbonyl': {   # Test F1: 0.7500, Test AUROC: 0.9123
            'hidden_channels': 128,
            'lr': 2e-3,
            'weight_decay': 0,
            'epochs': 1000
        },
        'house_triangle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
        'grid_triangle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
        'house_cycle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
    },

    'GIN_3layer': {
        'Benzene': {    # Test F1: 0.8957, Test AUROC: 0.9505
            'hidden_channels': 64,
            'lr': 1e-2,
            'weight_decay': 1e-3,   # 1e-3
            'epochs': 100
        },
        'Mutagenicity': {   # Test F1: 0.9928, Test AUROC: 0.9946
            'hidden_channels': 64,
            'lr': 1e-2,
            'weight_decay': 0,
            'epochs': 100
        },
        'AlkaneCarbonyl': {  # Test F1: 0.9427, Test AUROC: 0.9818
            'hidden_channels': 64,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 100
        },
        'FluorideCarbonyl': {   # Test F1: 0.7944, Test AUROC: 0.9472
            'hidden_channels': 128,
            'lr': 3e-3,
            'weight_decay': 1e-4,
            'epochs': 500
        },
        'house_triangle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
        'grid_triangle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
        'house_cycle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
    },

    'GAT_2layer': {
        'Benzene': {    # 
            'hidden_channels': 64,
            'lr': 1e-2,
            'weight_decay': 1e-3,   
            'epochs': 100
        },
        'Mutagenicity': {   # 
            'hidden_channels': 64,
            'lr': 1e-2,
            'weight_decay': 0,
            'epochs': 100
        },
        'AlkaneCarbonyl': {  # 
            'hidden_channels': 64,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 100
        },
        'FluorideCarbonyl': {   # 
            'hidden_channels': 128,
            'lr': 3e-3,
            'weight_decay': 1e-4,
            'epochs': 500
        },
        'house_triangle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
        'grid_triangle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
        'house_cycle': {    # 
            'hidden_channels': 32,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'epochs': 1000
        },
    }
}


def get_model(dataset, model_type, load_state=True):
    dataset_name = dataset.name
    if model_type == 'GIN_3layer':
        model = GIN_3layer(in_channels=dataset.num_features,
                           hidden_channels=hparams[model_type][dataset_name]['hidden_channels'],
                           out_channels=dataset.num_classes).to(device)

    elif model_type == 'GCN_3layer':
        model = GCN_3layer(in_channels=dataset.num_features,
                           hidden_channels=hparams[model_type][dataset_name]['hidden_channels'],
                           out_channels=dataset.num_classes).to(device)
    elif model_type == 'GAT_3layer':
        model = GAT_3layer(in_channels=dataset.num_features,
                           hidden_channels=hparams[model_type][dataset_name]['hidden_channels'],
                           out_channels=dataset.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams[model_type][dataset_name]['lr'],
                                 weight_decay=hparams[model_type][dataset_name]['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model_pth = os.path.join('gnn', 'model_weights', f'{dataset_name}_{model_type}.pth')
    model = train_model(dataset, model, optimizer, criterion, model_pth, epochs=hparams[model_type][dataset_name]['epochs'], load_state=load_state)

    print(f"GNN type: {model_type},\
          hidden_channels: {hparams[model_type][dataset_name]['hidden_channels']},\
          lr: {hparams[model_type][dataset_name]['lr']},\
          weight_decay: {hparams[model_type][dataset_name]['weight_decay']}")

    return model


if __name__ == "__main__":
    import itertools

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True

    # 仅搜索 GAT_3layer；可修改数据集名称以切换不同数据集
    model_type = 'GAT_2layer'   # 'GCN_3layer', 'GIN_3layer', 'GAT_2layer'
    dataset_name = 'FluorideCarbonyl'  # 可选：# Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, grid_triangle, house_cycle
    dataset = get_dataset(dataset_name, seed, device)

    # 搜索空间（适中规模，兼顾效果与时长）
    hidden_channels_grid = [32, 64, 128]
    lr_grid = [1e-2, 1e-3, 1e-4]
    weight_decay_grid = [0.0, 1e-3]
    default_epochs = 300

    # 记录最佳结果
    best_cfg = None
    best_val_f1 = -1.0
    best_test_metrics = None  # (f1, precision, recall, auprc, auroc, acc)

    # DataLoaders（每个配置训练时会在 train_model 内部重新创建）
    _, _ = dataset.get_train_loader(batch_size=128)
    val_loader, _ = dataset.get_val_loader()
    test_loader, _ = dataset.get_test_loader()

    print(f"Start HP search for {model_type} on {dataset_name} ...")
    for hidden_channels, lr, weight_decay in itertools.product(hidden_channels_grid, lr_grid, weight_decay_grid):
        # 构建模型与优化器
        model = GAT_3layer(in_channels=dataset.num_features,
                           hidden_channels=hidden_channels,
                           out_channels=dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # 为该组合单独设置权重保存路径，避免互相覆盖
        safe_lr = f"{lr:.0e}" if lr < 1e-2 else ("0.01" if abs(lr - 1e-2) < 1e-12 else f"{lr}")
        model_pth = os.path.join('gnn', 'model_weights', 'hparam_search', model_type, dataset_name,
                                 f"h{hidden_channels}_lr{safe_lr}_wd{weight_decay}.pth")

        print(f"\n=== Trial: hidden={hidden_channels}, lr={lr}, weight_decay={weight_decay} ===")
        # 训练：内部基于验证 F1 选择最佳 epoch 并加载其权重
        model = train_model(dataset, model, optimizer, criterion, model_pth, epochs=default_epochs, load_state=False)

        # 评估：以验证集 F1 作为选择标准
        from gnn.utils import test as eval_fn
        val_f1, val_prec, val_rec, val_auprc, val_auroc, val_acc = eval_fn(model, val_loader)
        print(f"Val -> F1: {val_f1:.4f}, AUROC: {val_auroc:.4f}, ACC: {val_acc:.4f}")

        # 同时记录测试集指标，便于最终报告
        test_f1, test_prec, test_rec, test_auprc, test_auroc, test_acc = eval_fn(model, test_loader)
        print(f"Test -> F1: {test_f1:.4f}, AUROC: {test_auroc:.4f}, ACC: {test_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_cfg = {
                'hidden_channels': hidden_channels,
                'lr': lr,
                'weight_decay': weight_decay,
                'epochs': default_epochs,
                'model_pth': model_pth
            }
            best_test_metrics = {
                'f1': test_f1,
                'precision': test_prec,
                'recall': test_rec,
                'auprc': test_auprc,
                'auroc': test_auroc,
                'accuracy': test_acc
            }

    print("\n===== Hyperparameter Search Summary =====")
    if best_cfg is None:
        print("No valid configuration evaluated.")
    else:
        print(f"Best (by Val F1) -> hidden={best_cfg['hidden_channels']}, lr={best_cfg['lr']}, "
              f"weight_decay={best_cfg['weight_decay']}, epochs={best_cfg['epochs']}")
        print(f"Saved weights: {best_cfg['model_pth']}")
        print("Best Test Metrics:")
        print(json.dumps(best_test_metrics, indent=2))
