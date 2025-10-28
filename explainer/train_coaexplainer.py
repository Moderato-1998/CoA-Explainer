from tqdm import tqdm
import torch
import numpy as np

from graphxai.utils import Explanation
from metrics.edge_exp_metrics import soft_mask_to_hard, fid_neg, fid_pos, jac_edge_max, jac_edge_all, faith_edge, sparsity_edge


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(explainer, train_loader, epoch, num_epochs, num_episodes_per_graph):
    explainer.actor.train()
    explainer.critic.train()
    explainer.state_gnn.train()

    epoch_actor_losses = []
    epoch_critic_losses = []
    epoch_rewards = []
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="graph", leave=False)

    for batch_idx, graph_data_batch in enumerate(train_loader_tqdm):
        graph_data = graph_data_batch.to(device)

        reward, _, _ = explainer.train_on_graph(graph_data, num_episodes_per_graph)
        epoch_rewards.append(reward)

        if len(explainer.replay_buffer) >= explainer.batch_size:
            updated, actor_l, critic_l = explainer.update_agents()
            if updated:
                epoch_actor_losses.append(actor_l)
                epoch_critic_losses.append(critic_l)
                train_loader_tqdm.set_postfix({"ActorL": f"{actor_l:.3f}", "CriticL": f"{critic_l:.3f}", "R": f"{reward:.2f}"})

    avg_epoch_actor_loss = np.mean(epoch_actor_losses)
    avg_epoch_critic_loss = np.mean(epoch_critic_losses)
    avg_epoch_reward = np.mean(epoch_rewards)
    return avg_epoch_actor_loss, avg_epoch_critic_loss, avg_epoch_reward


def evaluate_model(explainer, data_loader, exp_ls, epoch, num_epochs, evaluate_class):
    explainer.actor.eval()
    explainer.critic.eval()
    explainer.state_gnn.eval()

    correct = 0
    sparsity_score = 0.0
    raw_fidelity = 0.0
    exp_acc = 0.0
    num_exp_gt = 0.0

    loader_tqdm = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="graph", leave=False)
    with torch.no_grad():
        for idx, data_batch in enumerate(loader_tqdm):
            data = data_batch.to(device)
            total_r, raw_fid, sparsity_s, subgraph_pred_class, \
                original_pred_class, _, _, _, edge_mask = explainer.evaluate_graph(data)

            raw_fidelity += raw_fid
            sparsity_score += sparsity_s
            if original_pred_class != -1 and subgraph_pred_class == original_pred_class:
                correct += 1
            loader_tqdm.set_postfix({"ValFid": f"{raw_fid:.2f}", "ValSpar": f"{sparsity_s:.2f}", "R": f"{total_r:.2f}"})
            if data.y.item() in evaluate_class:
                num_exp_gt += 1
                exp = Explanation(
                    feature_imp=None,  # No feature importance - everything is one-hot encoded
                    node_imp=None,
                    edge_imp=edge_mask,
                )
                # print(f'exp_gt:{exp_ls[data.exp_key[0][0]][0].edge_imp}')
                # print(f'exp_pred:{exp.edge_imp}')
                JAC_edge = jac_edge_all(exp_ls[data.exp_key[0][0]], exp)
                exp_acc += JAC_edge
                # print(JAC_edge)

    num_graphs = len(data_loader)
    avg_acc = correct / num_graphs
    avg_sparsity = (sparsity_score / num_graphs)
    avg_raw_fidelity = raw_fidelity / num_graphs
    exp_acc = exp_acc / num_exp_gt

    return avg_acc, avg_sparsity, avg_raw_fidelity, exp_acc


def train_coaexp(train_loader, val_loader, evl_exp_ls,
                  explainer, num_epochs=20, num_episodes_per_graph=1,
                  save_path=None, evaluate_class=[1]):

    best_val_combined_metric = 0
    best_model_state = {'actor': None, 'critic': None, 'state_gnn': None}

    info = []

    for epoch in range(num_epochs):
        # Training
        avg_epoch_actor_loss, avg_epoch_critic_loss, avg_epoch_reward = train_loop(explainer, train_loader, epoch, num_epochs, num_episodes_per_graph)

        # Validation
        avg_acc, avg_sparsity, avg_raw_fidelity, exp_acc = evaluate_model(explainer, val_loader, evl_exp_ls, epoch, num_epochs, evaluate_class)

        # Save the best model for sparsity and explanation accuracy
        # combined_metric =  0.35*avg_sparsity + exp_acc
        combined_metric = avg_sparsity+exp_acc+avg_acc
        if combined_metric >= best_val_combined_metric:
            best_val_combined_metric = combined_metric
            best_model_state['actor'] = explainer.actor.state_dict().copy()
            best_model_state['critic'] = explainer.critic.state_dict().copy()
            best_model_state['state_gnn'] = explainer.state_gnn.state_dict().copy()
            if save_path:
                explainer.save_model(save_path)
                print(f"Saved model: E {epoch+1}/{num_epochs}")

        epoch_info = (f"E {epoch+1}/{num_epochs}: Train ALoss {avg_epoch_actor_loss:.3f}, CLoss {avg_epoch_critic_loss:.3f}, R {avg_epoch_reward:.3f} | "
              f"Subgraph Acc {avg_acc:.2f}, Fid {avg_raw_fidelity:.3f}, Sparsity {avg_sparsity:.3f} (score), Combined {combined_metric:.2f}, GEA {exp_acc:.2f}")
        print(epoch_info)
        info.append(epoch_info)
    
    return info