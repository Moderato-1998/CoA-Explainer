import torch
import os
from graphxai.datasets import AlkaneCarbonyl, Mutagenicity, MUTAG, Benzene, FluorideCarbonyl
from graphxai.datasets import BAHouse, BADiamond, BAWheel, BACycle
from data.ba2motif import BA2Motif

device = "cuda" if torch.cuda.is_available() else "cpu"


def print_dataset_info(dataset):
    print("=" * 20, "Dataset  Information", "=" * 20)
    print(f"Dataset: {dataset.name}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of graphs: {len(dataset.graphs)}")    
    print(f'Average number of nodes: {dataset.ave_nodes}')
    print(f'Average number of edges: {dataset.ave_edges}')
    print(f"Train size: {len(dataset.train_index)}")
    print(f"Validation size: {len(dataset.val_index)}")
    print(f"Test size: {len(dataset.test_index)}")
    print(f'Shuffle:{ dataset.shuffle}')
    print(f"Device: {device}")
    print("=" * 60)


def get_dataset(name, seed, device):

    name = name.lower()
    if name == 'Benzene'.lower():
        dataset = Benzene(seed=seed, device=device)

    elif name == 'AlkaneCarbonyl'.lower():
        dataset = AlkaneCarbonyl(seed=seed, device=device)

    elif name == 'Mutagenicity'.lower():
        dataset = Mutagenicity(root=os.path.join('data'), seed=seed, device=device)

    elif name == 'MUTAG'.lower():
        dataset = MUTAG(root=os.path.join('data'), seed=seed, device=device)

    elif name == 'FluorideCarbonyl'.lower():
        dataset = FluorideCarbonyl(seed=seed, device=device)

    elif name == 'house_triangle'.lower():
        dataset = BA2Motif(shape1='house', shape2='triangle', seed=seed, device=device)

    elif name == 'triangle_grid'.lower():
        dataset = BA2Motif(shape1='triangle', shape2='grid', seed=seed, device=device)

    elif name == 'house_cycle'.lower():
        dataset = BA2Motif(shape1='house', shape2='cycle', seed=seed, device=device)

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    print_dataset_info(dataset)
    return dataset


def correct(g, model):

    model.eval()
    with torch.no_grad():
        pred, _ = model(g.x, g.edge_index, g.batch)
        pred_class = torch.argmax(pred, dim=-1).item()
        if pred_class == g.y.item():
            return True
        else:
            return False


def filter_correct_data(dataset, model):

    correcr_train_index = []
    correcr_val_index = []
    correcr_test_index = []

    for idx in dataset.train_index:
        g = dataset.graphs[idx].to(device)
        if correct(g, model):
            correcr_train_index.append(idx)

    for idx in dataset.val_index:
        g = dataset.graphs[idx].to(device)
        if correct(g, model):
            correcr_val_index.append(idx)

    for idx in dataset.test_index:
        g = dataset.graphs[idx].to(device)
        if correct(g, model):
            correcr_test_index.append(idx)

    return correcr_train_index, correcr_val_index, correcr_test_index
