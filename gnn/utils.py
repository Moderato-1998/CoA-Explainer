import torch
from tqdm import tqdm
import numpy as np
import os

from torch_geometric.loader import DataLoader

import sklearn.metrics as metrics
from sklearn.metrics import f1_score, precision_score, recall_score

from gnn.models import GCN_3layer, GIN_3layer, GAT_3layer


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, data_loader: DataLoader):
    model.train()
    for data in data_loader:  # Iterate in batches over the training dataset.
        out, _ = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model: torch.nn.Module, data_loader: DataLoader):
    with torch.no_grad():
        model.eval()
        GT = np.zeros(len(data_loader))
        preds = np.zeros(len(data_loader))
        probas = np.zeros(len(data_loader))

        i = 0
        for data in data_loader:
            out, _ = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).item()
            GT[i] = data.y.item()
            preds[i] = pred

            probas[i] = out.softmax(dim=1).squeeze()[1].detach().clone().cpu().numpy()

            i += 1

        f1 = f1_score(GT, preds)
        precision = precision_score(GT, preds)
        recall = recall_score(GT, preds)
        auprc = metrics.average_precision_score(GT, probas)
        auroc = metrics.roc_auc_score(GT, probas)
        accuracy = metrics.accuracy_score(GT, preds)

        return f1, precision, recall, auprc, auroc, accuracy


def train_model(dataset, model, optimizer, criterion, model_pth, epochs=100, load_state=False):

    if load_state and os.path.exists(model_pth):
        model.load_state_dict(torch.load(model_pth, map_location=device))
        print(f"Loaded gnn from {model_pth}")
        return model

    else:
        optimizer = optimizer
        criterion = criterion

        train_loader, _ = dataset.get_train_loader(batch_size=128)
        val_loader, _ = dataset.get_val_loader()
        test_loader, _ = dataset.get_test_loader()

        best_f1 = 0
        best_model_state = None

        data_name = dataset.name
        print(f"Training GNN on the {data_name} train set...")
        for epoch in range(0, epochs):
            train(model, optimizer, criterion, train_loader)
            f1, prec, rec, auprc, auroc, acc = test(model, val_loader)
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()
            print(f'Epoch: {epoch:03d}, Val F1: {f1:.4f}, Val AUROC: {auroc:.4f}, ACC: {acc:.4f}')

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            os.makedirs(os.path.dirname(model_pth), exist_ok=True)
            torch.save(best_model_state, model_pth)

        print(f"Testing the best model on the {data_name} test set...")
        f1, precision, recall, auprc, auroc, acc = test(model, test_loader)
        print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}, Test ACC: {acc:.4f}')

        return model
