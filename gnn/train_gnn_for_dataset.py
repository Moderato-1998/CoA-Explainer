import os
import torch
import json
from graphxai.datasets import AlkaneCarbonyl, Mutagenicity, Benzene, FluorideCarbonyl
from graphxai.datasets import BAHouse, BADiamond, BAWheel, BACycle
from gnn.utils import train_model
from gnn.models import GCN_3layer, GIN_3layer, GAT_2layer
import random
import numpy as np
import torch_geometric
from data.utils import get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


hparams = {
    "GCN_3layer": {
        "Benzene": {  # Test F1: 0.8711, Test AUROC: 0.9549, Test ACC: 0.8750
            "hidden_channels": 64,
            "lr": 5e-3,
            "weight_decay": 0,
            "epochs": 1000,
        },
        "Mutagenicity": {  # Test F1: 0.9819, Test AUROC: 0.9975, Test ACC: 0.9859
            "hidden_channels": 64,
            "lr": 1e-2,
            "weight_decay": 0,
            "epochs": 1000,
        },
        "AlkaneCarbonyl": {  # Test F1: 0.9172, Test AUROC: 0.9764, Test ACC: 0.9422
            "hidden_channels": 64,
            "lr": 3e-3,
            "weight_decay": 0,
            "epochs": 1000,
        },
        "FluorideCarbonyl": {  # Test F1: 0.7143, Test AUROC: 0.9205, Test ACC: 0.8939
            "hidden_channels": 128,
            "lr": 2e-3,
            "weight_decay": 0,
            "epochs": 1000,
        },
        "house_triangle": {  # Test F1: 1.0000, Test AUROC: 1.0000, Test ACC: 1.0000
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "grid_triangle": {  # Test F1: 1.0000, Test AUROC: 1.0000, Test ACC: 1.0000
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "house_cycle": {  # Test F1: 0.9952, Test AUROC: 1.0000, Test ACC: 0.9950
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
    },
    "GIN_3layer": {
        "Benzene": {  # Test F1: 0.8727, Test AUROC: 0.9553, Test ACC: 0.8738
            "hidden_channels": 64,
            "lr": 1e-2,
            "weight_decay": 1e-3,
            "epochs": 1000,
        },
        "Mutagenicity": {  # Test F1: 0.9786, Test AUROC: 0.9895, Test ACC: 0.9831
            "hidden_channels": 64,
            "lr": 1e-2,
            "weight_decay": 0,
            "epochs": 1000,
        },
        "AlkaneCarbonyl": {  # Test F1: 0.8767, Test AUROC: 0.9668, Test ACC: 0.9200
            "hidden_channels": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 300,
        },
        "FluorideCarbonyl": {  # Test F1: 0.7917, Test AUROC: 0.9420, Test ACC: 0.9308
            "hidden_channels": 128,
            "lr": 3e-3,
            "weight_decay": 1e-4,
            "epochs": 300,
        },
        "house_triangle": {  # Test F1: 0.9754, Test AUROC: 0.9994, Test ACC: 0.9750
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "triangle_grid": {  # Test F1: 1.0000, Test AUROC: 1.0000, Test ACC: 1.0000
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "house_cycle": {  # Test F1: 0.9561, Test AUROC: 0.9938, Test ACC: 0.9550
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
    },
    "GAT_2layer": {
        "Benzene": {  # Test F1: 0.8353, Test AUROC: 0.9092, Test ACC: 0.8304
            "hidden_channels": 64,
            "lr": 0.003,
            "weight_decay": 0.0001,
            "epochs": 1000,
        },
        "Mutagenicity": {  # Test F1: 0.9682, Test AUROC: 0.9927, Test ACC: 0.9746
            "hidden_channels": 64,
            "lr": 0.003,
            "weight_decay": 0,
            "epochs": 1000,
        },
        "AlkaneCarbonyl": {  # Test F1: 0.9161, Test AUROC: 0.9757, Test ACC: 0.9422
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "FluorideCarbonyl": {  # Test F1: 0.2698, Test AUROC: 0.8153, Test ACC: 0.8408
            "hidden_channels": 128,
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "epochs": 1000,
        },
        "house_triangle": {  # Test F1: 1.0000, Test AUROC: 1.0000, Test ACC: 1.0000
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "grid_triangle": {  # Test F1: 1.0000, Test AUROC: 1.0000, Test ACC: 1.0000
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
        "house_cycle": {  # Test F1: 1.0000, Test AUROC: 1.0000, Test ACC: 1.0000
            "hidden_channels": 32,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1000,
        },
    },
}


def get_model(dataset, model_type, load_state=True):
    dataset_name = dataset.name
    if model_type == "GIN_3layer":
        model = GIN_3layer(
            in_channels=dataset.num_features,
            hidden_channels=hparams[model_type][dataset_name]["hidden_channels"],
            out_channels=dataset.num_classes,
        ).to(device)

    elif model_type == "GCN_3layer":
        model = GCN_3layer(
            in_channels=dataset.num_features,
            hidden_channels=hparams[model_type][dataset_name]["hidden_channels"],
            out_channels=dataset.num_classes,
        ).to(device)
    elif model_type == "GAT_2layer":
        model = GAT_2layer(
            in_channels=dataset.num_features,
            hidden_channels=hparams[model_type][dataset_name]["hidden_channels"],
            out_channels=dataset.num_classes,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams[model_type][dataset_name]["lr"],
        weight_decay=hparams[model_type][dataset_name]["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model_pth = os.path.join("gnn", "model_weights", f"{dataset_name}_{model_type}.pth")
    model = train_model(
        dataset,
        model,
        optimizer,
        criterion,
        model_pth,
        epochs=hparams[model_type][dataset_name]["epochs"],
        load_state=load_state,
    )

    print(
        f"GNN type: {model_type},\
          hidden_channels: {hparams[model_type][dataset_name]['hidden_channels']},\
          lr: {hparams[model_type][dataset_name]['lr']},\
          weight_decay: {hparams[model_type][dataset_name]['weight_decay']}"
    )

    return model


if __name__ == "__main__":
    # train and save model for task
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True

    model_type = "GIN_3layer"  # 'GCN_3layer', 'GIN_3layer', 'GAT_2layer'

    dataset = get_dataset(
        "FluorideCarbonyl", seed, device
    )  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle

    model = get_model(dataset, model_type, load_state=True)

    from gnn.utils import test

    test_loader, _ = dataset.get_test_loader()
    f1, precision, recall, auprc, auroc, accuracy = test(model, test_loader)
    print(f"Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}, Test ACC: {accuracy:.4f}")
