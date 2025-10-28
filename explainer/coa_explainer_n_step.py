import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch
import torch.nn.functional as F
import os
import math


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits


class Critic(nn.Module):
    def __init__(self, global_state_dim, global_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # If there are no global actions, adjust input dimension
        input_dim = global_state_dim + global_action_dim if global_action_dim > 0 else global_state_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, global_actions):
        if global_actions is not None and global_actions.numel() > 0:
             x = torch.cat([global_state, global_actions], dim=-1)
        else:
             x = global_state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class CoAExplainer:
    def __init__(self, gnn_model, num_node_features, max_nodes, node_embedding_dim,
                 actor_lr=1e-4, critic_lr=3e-4, gamma=0.99, tau=0.01,
                 device='cpu',
                 batch_size=64, buffer_capacity=10000,
                 exploration_noise_std: float = 0.05,):
        self.device = device

        self.gnn_model_to_explain = gnn_model   # Target GNN model to explain
        self.gnn_model_to_explain.eval()

        self.state_gnn = copy.deepcopy(gnn_model)  # State GNN for actor input
        self.state_gnn.train()  # state_gnn parameters are part of actor_optimizer

        self.num_node_features = num_node_features
        self.node_embedding_dim = node_embedding_dim
        self.max_nodes = max_nodes

        self.agent_state_dim = self.node_embedding_dim
        self.agent_action_dim = 1

        self.actor = Actor(self.agent_state_dim, self.agent_action_dim).to(device)
        self.target_actor = Actor(self.agent_state_dim, self.agent_action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        actor_params = list(self.actor.parameters()) + list(self.state_gnn.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr)

        self.critic_global_state_dim = self.max_nodes * self.agent_state_dim
        self.critic_global_action_dim = self.max_nodes * self.agent_action_dim

        self.critic = Critic(self.critic_global_state_dim, self.critic_global_action_dim).to(device)
        self.target_critic = Critic(self.critic_global_state_dim, self.critic_global_action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Learning rate schedulers removed from public API. If you need schedulers,
        # reintroduce them in training script or via an explicit setter.
        # self.actor_lr_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
        # self.critic_lr_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = []
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self._train_call_count = 0
        # Exploration noise std for action logits during training
        self.noise_std = float(exploration_noise_std)
        # Multi-step episode length (>=1). Keep default=1 to preserve previous behavior.
        # Set to >1 to enable true multi-step cooperative pruning (MADDPG-style)
        self.steps_per_episode = 1

    def _pad_tensor(self, tensor, max_len, dim=0):
        current_len = tensor.size(dim)
        if current_len == max_len:
            return tensor
        elif current_len > max_len:
            indices = torch.arange(max_len, device=tensor.device)
            return tensor.index_select(dim, indices)
        else:
            padding_size = max_len - current_len
            pad_shape = list(tensor.shape)
            pad_shape[dim] = padding_size
            padding = torch.zeros(
                pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=dim)

    def _get_padded_global_states_actions(self, agent_states, agent_actions):
        # (num_nodes, agent_state_dim) -> (max_nodes, agent_state_dim)
        padded_states = self._pad_tensor(agent_states, self.max_nodes, dim=0)
        padded_actions = self._pad_tensor(agent_actions, self.max_nodes, dim=0)

        # (max_nodes * agent_state_dim)
        flat_padded_states = padded_states.reshape(-1)
        flat_padded_actions = padded_actions.reshape(-1)

        # Add batch dim: (1, max_nodes * dim)
        return flat_padded_states.unsqueeze(0), flat_padded_actions.unsqueeze(0)

    def select_actions(self, agent_states_batch, add_noise=False):
        actions_logits = self.actor(agent_states_batch)
        if add_noise:
            noise = torch.randn_like(actions_logits) * self.noise_std
            actions_logits = actions_logits + noise
        return actions_logits

    def _actions_to_edge_mask(self, graph_data, node_action_logits, use_gumbel_softmax=False, temperature=1.0):
        edge_index = graph_data.edge_index
        num_edges = edge_index.size(1)

        if num_edges == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device), torch.empty(0, dtype=torch.float, device=self.device)

        src_importances = node_action_logits[edge_index[0]]
        tgt_importances = node_action_logits[edge_index[1]]
        edge_logits = (src_importances + tgt_importances) / 2.0
        edge_logits = edge_logits.squeeze(-1)
        edge_probs = torch.sigmoid(edge_logits)

        # Remove fixed/top-k style min-edge heuristics so size is truly adaptive per-instance
        if self.actor.training:
            edge_mask = torch.bernoulli(edge_probs).bool()
        else:
            edge_mask = (edge_probs > 0.5).bool()

        if self.actor.training and num_edges > 0 and edge_mask.sum() == 0:
            max_prob_edge_idx = torch.argmax(edge_probs)
            edge_mask[max_prob_edge_idx] = True

        return edge_mask, edge_logits

    def _calculate_reward(self, original_graph_data, subgraph_data, original_pred_logits, pred_target_class_idx):
        """
        MDL-style reward (no manual sparsity/fidelity weights):
          total_reward = log p(y_orig | subgraph) - log C(M, K)
        where M is the number of original edges and K is the number of edges kept.
        This intrinsically trades off fidelity and sparsity per-instance.
        Returns: total_reward, raw_fidelity_term, normalized_sparsity_score
        """
        # Handle degenerate cases
        if subgraph_data.num_nodes == 0:
            # No nodes -> no predictive power; reward very low, sparsity high
            raw_fidelity_term = -10.0
            sparsity_score = 1.0
            return raw_fidelity_term - 0.0, raw_fidelity_term, sparsity_score

        # Compute fidelity term: log prob of original predicted class under subgraph
        try:
            with torch.no_grad():
                # Subgraph prediction: use node-induced subgraph to better align with GCN behavior on small graphs
                # Build node-induced subgraph from subgraph_data.edge_index
                from torch_geometric.utils import subgraph as tg_subgraph
                masked_edge_index = subgraph_data.edge_index
                if masked_edge_index.numel() > 0:
                    nodes_exp = torch.unique(masked_edge_index)
                    e_idx_sub, _ = tg_subgraph(nodes_exp, original_graph_data.edge_index.to(self.device), relabel_nodes=True)
                    x_sub_nodes = original_graph_data.x.to(self.device)[nodes_exp]
                    b_sub_nodes = torch.zeros(x_sub_nodes.size(0), dtype=torch.long, device=self.device)
                    subgraph_pred_logits, _ = self.gnn_model_to_explain(x_sub_nodes, e_idx_sub, b_sub_nodes)
                else:
                    subgraph_pred_logits, _ = self.gnn_model_to_explain(
                        subgraph_data.x.to(self.device),
                        subgraph_data.edge_index.to(self.device),
                        torch.zeros(subgraph_data.x.size(0), dtype=torch.long, device=self.device)
                    )
                # Full-graph baseline (already available as original_pred_logits)
                full_log_probs = F.log_softmax(original_pred_logits, dim=-1)
                sub_log_probs = F.log_softmax(subgraph_pred_logits, dim=-1)
            # Differential fidelity: improves stability across models
            raw_fidelity_term = (sub_log_probs[0, pred_target_class_idx] - full_log_probs[0, pred_target_class_idx]).item()
        except Exception:
            raw_fidelity_term = -10.0

        # Combinatorial description length of chosen edges (in nats)
        M = int(original_graph_data.num_edges) if hasattr(original_graph_data, 'num_edges') else original_graph_data.edge_index.size(1)
        K = int(subgraph_data.num_edges) if hasattr(subgraph_data, 'num_edges') else subgraph_data.edge_index.size(1)

        if M <= 0:
            mdl_penalty = 0.0
            sparsity_score = 1.0
        else:
            K_clamped = max(0, min(K, M))
            # log C(M, K) = lgamma(M+1) - lgamma(K+1) - lgamma(M-K+1)
            # compute on CPU via math.lgamma to avoid CUDA NVRTC deps
            mdl_logC = (math.lgamma(M + 1.0)
                           - math.lgamma(K_clamped + 1.0)
                           - math.lgamma(M - K_clamped + 1.0))
            # Normalize by M for scale stability across graphs of different sizes
            mdl_logC_norm = mdl_logC / max(1.0, float(M))

            # Additional param-free penalty: KL(p || p0) with p0 = 1 / max(2, sqrt(M))
            # This biases towards sparsity while remaining adaptive to graph size
            M_float = float(M)
            p = K_clamped / M_float
            eps = 1e-6
            p = min(max(p, eps), 1.0 - eps)
            p0 = 1.0 / max(2.0, math.sqrt(M_float))
            p0 = min(max(p0, eps), 1.0 - eps)
            kl_pp0 = p * math.log(p / p0) + (1.0 - p) * math.log((1.0 - p) / (1.0 - p0))

            mdl_penalty = mdl_logC_norm + kl_pp0

            sparsity_score = 1.0 - (K_clamped / M_float)
            sparsity_score = max(0.0, min(1.0, sparsity_score))
        # Gate the sparsity penalty by the sign/magnitude of fidelity gain on small graphs
        # Intuition: when Δloglik<0 (removing edges hurts prediction), let agent focus on recovering fidelity;
        # as Δ>=0 increases, gradually introduce sparsity pressure.
        # Ensure a minimal penalty even when Δ≈0 or Δ<0 to avoid degenerate keep-all-edges solutions on small graphs
        # Use graph-size adaptive epsilon so that eps is not vanishing: eps ~ 1/max(10, M)
        eps_gate = 1.0 / max(10.0, float(M))
        delta = float(raw_fidelity_term)
        gate = (max(0.0, delta) + eps_gate) / (abs(delta) + eps_gate)
        total_reward = raw_fidelity_term - gate * mdl_penalty
        return total_reward, raw_fidelity_term, sparsity_score

    def _get_subgraph(self, original_graph_data, edge_mask):
        original_graph_data = original_graph_data.to(self.device)
        subgraph_data = original_graph_data.clone()  # Clones x, num_nodes etc.

        if original_graph_data.edge_index is None or original_graph_data.edge_index.size(1) == 0:
            subgraph_data.edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device)
            if hasattr(subgraph_data, 'edge_attr') and subgraph_data.edge_attr is not None:
                subgraph_data.edge_attr = torch.empty((0, subgraph_data.edge_attr.size(1)),
                                                      dtype=subgraph_data.edge_attr.dtype, device=self.device)
            subgraph_data.num_edges = 0
            return subgraph_data

        if edge_mask.size(0) != original_graph_data.edge_index.size(1):
            raise ValueError(
                f"edge_mask size ({edge_mask.size(0)}) != num_edges ({original_graph_data.edge_index.size(1)})")

        subgraph_data.edge_index = original_graph_data.edge_index[:, edge_mask]
        if hasattr(original_graph_data, 'edge_attr') and original_graph_data.edge_attr is not None:
            if subgraph_data.edge_index.size(1) > 0:  # if there are edges left
                subgraph_data.edge_attr = original_graph_data.edge_attr[edge_mask]
            else:  # no edges left, empty edge_attr
                subgraph_data.edge_attr = torch.empty((0, original_graph_data.edge_attr.size(1)),
                                                      dtype=original_graph_data.edge_attr.dtype,
                                                      device=self.device)

        # Update num_edges attribute for the subgraph
        subgraph_data.num_edges = subgraph_data.edge_index.size(1)
        return subgraph_data

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done, num_nodes_for_experience):
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)

        # next_state is now expected to be a tensor (padded global state for S')
        # It's already detached in train_on_graph before padding
        # Assuming next_state is already correctly (1, max_nodes * dim)
        next_state_to_store = next_state

        self.replay_buffer.append((state,
                                   action,
                                   torch.tensor([reward], device=self.device),
                                   next_state_to_store,  # Should be a tensor, not None
                                   torch.tensor([done], device=self.device),
                                   torch.tensor(
                                       [num_nodes_for_experience], dtype=torch.long, device=self.device)
                                   ))

    def update_agents(self):
        if len(self.replay_buffer) < self.batch_size:
            return False, None, None

        indices = np.random.choice(
            len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states_b, actions_b, rewards_b, next_states_b, dones_b, num_nodes_b = zip(
            *batch)

        states_batch = torch.cat(states_b)
        actions_batch = torch.cat(actions_b)
        rewards_batch = torch.cat(rewards_b).unsqueeze(1)
        # Reward normalization to reduce variance (helps with batch=1 graphs and GCN training)
        with torch.no_grad():
            r_mean = rewards_batch.mean()
            r_std = rewards_batch.std()
        rewards_batch = (rewards_batch - r_mean) / (r_std + 1e-6)
        next_states_batch = torch.cat(next_states_b)  # S' batch
        dones_batch = torch.cat(dones_b).unsqueeze(1)
        # num_nodes for S and S' (since node set doesn't change)
        num_nodes_batch = torch.cat(num_nodes_b)

        with torch.no_grad():
            next_target_actions_list = []
            for i in range(self.batch_size):
                current_num_nodes = num_nodes_batch[i].item()
                if current_num_nodes == 0:
                    next_action_unpadded = torch.empty(
                        (0, self.agent_action_dim), device=self.device)
                else:
                    # Unpad S' using num_nodes
                    next_state_padded_flat = next_states_batch[i]
                    next_state_padded = next_state_padded_flat.view(
                        self.max_nodes, self.agent_state_dim)
                    next_state_unpadded = next_state_padded[:current_num_nodes, :]
                    next_action_unpadded = self.target_actor(
                        next_state_unpadded)  # Target actor acts on S'

                next_action_padded = self._pad_tensor(
                    next_action_unpadded, self.max_nodes, dim=0)
                next_target_actions_list.append(next_action_padded.view(-1))
            next_actions_target_batch = torch.stack(next_target_actions_list)

            Q_targets_next = self.target_critic(
                next_states_batch, next_actions_target_batch)
            Q_targets = rewards_batch + \
                (self.gamma * Q_targets_next * (1 - dones_batch.float()))

        current_Q_values = self.critic(
            states_batch, actions_batch)  # Critic acts on (S, A)
        critic_loss = F.mse_loss(current_Q_values, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_pred_actions_list = []
        for i in range(self.batch_size):
            current_num_nodes = num_nodes_batch[i].item()
            if current_num_nodes == 0:
                action_unpadded = torch.empty(
                    (0, self.agent_action_dim), device=self.device)
            else:
                # Unpad S using num_nodes
                state_padded_flat = states_batch[i]
                state_padded = state_padded_flat.view(
                    self.max_nodes, self.agent_state_dim)
                # IMPORTANT: For state_gnn to train, state_unpadded should be recomputed here from original graph data
                # stored in buffer, not from the detached states_batch.
                # For now, sticking to current detached state usage for simplicity, meaning state_gnn isn't trained by actor_loss this way.
                # This is a limitation of the current code structure if state_gnn is intended to be learned by actor_optimizer.
                # If state_gnn is part of actor_params, its params are updated, but grads from actor loss don't reach it
                # if states_batch are detached.
                state_unpadded = state_padded[:current_num_nodes, :]
                action_unpadded = self.actor(state_unpadded)  # Actor acts on S

            action_padded = self._pad_tensor(
                action_unpadded, self.max_nodes, dim=0)
            actor_pred_actions_list.append(action_padded.view(-1))
        actor_actions_pred_batch = torch.stack(actor_pred_actions_list)

        # Actor loss: depends on Q(S, A_pred_by_actor)
        actor_loss = -self.critic(states_batch,
                                  actor_actions_pred_batch).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Clip gradients for actor parameters (MLP and state_gnn)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if self.state_gnn:  # Also clip state_gnn if it's part of actor_params and being trained
            torch.nn.utils.clip_grad_norm_(self.state_gnn.parameters(), 1.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return True, actor_loss.item(), critic_loss.item()

    def train_on_graph(self, graph_data, num_episodes_per_graph=1):
        self._train_call_count += 1
        if graph_data.num_nodes == 0:
            return 0, 0, 0

        # Get original prediction (from model to be explained)
        with torch.no_grad():
            original_pred_logits, _ = self.gnn_model_to_explain(
                graph_data.x.to(self.device),
                graph_data.edge_index.to(self.device),
                torch.zeros(graph_data.num_nodes,
                            dtype=torch.long, device=self.device)
            )
        pred_target_class_idx = torch.argmax(
            original_pred_logits, dim=-1).item()

        total_rewards_graph = []
        raw_fidelity_rewards_graph = []
        sparsity_scores_graph = []

        for episode in range(num_episodes_per_graph):
            # Multi-step episode: iteratively prune edges
            current_graph = graph_data
            # Track number of nodes once (unchanged across pruning in current implementation)
            num_nodes_in_graph = current_graph.num_nodes

            for step in range(max(1, int(self.steps_per_episode))):
                # 1) Compute embeddings on current graph (state S_t)
                #    state_gnn is in train() mode; we keep autograd here, but will detach before storing in replay
                _, node_embeddings_t = self.state_gnn(
                    current_graph.x.to(self.device),
                    current_graph.edge_index.to(self.device),
                    torch.zeros(current_graph.num_nodes, dtype=torch.long, device=self.device)
                )

                # 2) Actor selects actions per node
                action_logits_per_node = self.select_actions(node_embeddings_t, add_noise=True)

                # 3) Build next subgraph by masking current_graph edges
                edge_mask_t, _ = self._actions_to_edge_mask(current_graph, action_logits_per_node)
                next_graph = self._get_subgraph(current_graph, edge_mask_t)

                # Ensure subgraph attributes
                if next_graph.x is None:
                    next_graph.x = torch.empty((0, self.num_node_features), device=self.device)
                if next_graph.edge_index is None:
                    next_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                if not hasattr(next_graph, 'batch') or next_graph.batch is None:
                    next_graph.batch = torch.zeros(next_graph.x.size(0), dtype=torch.long, device=self.device)
                if not hasattr(next_graph, 'num_edges'):
                    next_graph.num_edges = next_graph.edge_index.size(1)

                # 4) Reward relative to original prediction (dense reward per step)
                reward_t, raw_fidelity_t, sparsity_score_t = self._calculate_reward(
                    graph_data, next_graph, original_pred_logits, pred_target_class_idx
                )
                total_rewards_graph.append(reward_t)
                raw_fidelity_rewards_graph.append(raw_fidelity_t)
                sparsity_scores_graph.append(sparsity_score_t)

                # 5) Build (S_t, A_t) padded for replay (detached)
                state_t_unpadded = node_embeddings_t.detach()
                actions_t_unpadded = action_logits_per_node.detach()
                state_t_padded, actions_t_padded = self._get_padded_global_states_actions(
                    state_t_unpadded, actions_t_unpadded
                )

                # 6) Compute S_{t+1} on next_graph (no grad for target estimation)
                with torch.no_grad():
                    _, node_embeddings_tp1 = self.state_gnn(
                        next_graph.x.to(self.device),
                        next_graph.edge_index.to(self.device),
                        torch.zeros(next_graph.num_nodes, dtype=torch.long, device=self.device)
                    )
                next_state_unpadded = node_embeddings_tp1
                dummy_actions_for_pad = torch.zeros((next_state_unpadded.size(0), self.agent_action_dim), device=self.device)
                next_state_padded, _ = self._get_padded_global_states_actions(next_state_unpadded, dummy_actions_for_pad)

                # 7) done flag at final step or when graph becomes edgeless
                done_flag = (step == max(1, int(self.steps_per_episode)) - 1) or (next_graph.num_edges == 0)
                self.store_experience(state_t_padded, actions_t_padded,
                                      reward_t, next_state_padded, done_flag, num_nodes_in_graph)

                # 8) Transition to next graph
                current_graph = next_graph
                if done_flag:
                    break

        avg_total_reward = np.mean(
            total_rewards_graph) if total_rewards_graph else 0
        avg_raw_fidelity = np.mean(
            raw_fidelity_rewards_graph) if raw_fidelity_rewards_graph else 0
        avg_sparsity_score = np.mean(
            sparsity_scores_graph) if sparsity_scores_graph else 0

        return avg_total_reward, avg_raw_fidelity, avg_sparsity_score

    def evaluate_graph(self, graph_data):
        self.actor.eval()
        self.state_gnn.eval()

        graph_data = graph_data.to(self.device)
        if graph_data.num_nodes == 0:
            empty_subgraph = Data(x=torch.empty((0, self.num_node_features), device=self.device),
                                  edge_index=torch.empty(
                                      (2, 0), dtype=torch.long, device=self.device),
                                  num_nodes=0)
            empty_subgraph.num_edges = 0
            empty_mask = torch.empty(0, dtype=torch.bool, device=self.device)
            return 0.0, -5.0, 1.0, -1, -1, 0, 0, empty_subgraph, empty_mask

        with torch.no_grad():
            original_pred_logits, _ = self.gnn_model_to_explain(
                graph_data.x, graph_data.edge_index,
                torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device))
            original_pred_class = torch.argmax(
                original_pred_logits, dim=-1).item()

            _, node_embeddings_for_state_eval = self.state_gnn(
                graph_data.x, graph_data.edge_index,
                torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device))

        agent_states_for_eval = node_embeddings_for_state_eval
        action_logits_per_node = self.select_actions(
            agent_states_for_eval, add_noise=False)

        # Use deterministic mask for evaluation
        temp_training_state = self.actor.training
        # Ensure use_gumbel_softmax=False or bernoulli is not used if it checks actor.training
        self.actor.eval()
        edge_mask, edge_probs = self._actions_to_edge_mask(
            graph_data, action_logits_per_node, use_gumbel_softmax=False)
        self.actor.train(temp_training_state)  # Restore original mode

        subgraph_data = self._get_subgraph(graph_data, edge_mask)

        if subgraph_data.num_edges == 0 and graph_data.num_edges > 0:  # If subgraph is empty but original was not
            # Fallback: try to keep a few more edges if eval results in empty subgraph
            # print("Eval resulted in empty subgraph, trying lower threshold for mask.")
            sorted_probs, sorted_indices = torch.sort(
                edge_probs, descending=True)
            # Keep 10% or at least 1, max 10
            num_edges_to_keep = min(
                max(1, int(graph_data.num_edges * 0.1)), 10)

            fallback_mask = torch.zeros_like(edge_probs, dtype=torch.bool)
            if num_edges_to_keep > 0 and len(sorted_indices) >= num_edges_to_keep:
                fallback_mask[sorted_indices[:num_edges_to_keep]] = True
                subgraph_data = self._get_subgraph(graph_data, fallback_mask)

        # Ensure subgraph attributes for reward calculation and prediction
        if subgraph_data.x is None:
            subgraph_data.x = torch.empty(
                (0, self.num_node_features), device=self.device)
        if subgraph_data.edge_index is None:
            subgraph_data.edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device)
        if not hasattr(subgraph_data, 'batch') or subgraph_data.batch is None:
            subgraph_data.batch = torch.zeros(subgraph_data.x.size(
                0), dtype=torch.long, device=self.device)
        if not hasattr(subgraph_data, 'num_edges'):
            subgraph_data.num_edges = subgraph_data.edge_index.size(1)

        total_reward, raw_fidelity, sparsity_score = self._calculate_reward(
            graph_data, subgraph_data, original_pred_logits, original_pred_class)

        subgraph_pred_class = -1
        if subgraph_data.num_nodes > 0:  # Can predict if nodes exist, even if no edges
            try:
                with torch.no_grad():
                    subgraph_pred_logits, _ = self.gnn_model_to_explain(
                        subgraph_data.x, subgraph_data.edge_index, subgraph_data.batch)
                subgraph_pred_class = torch.argmax(
                    subgraph_pred_logits, dim=-1).item()
            except Exception as e:
                subgraph_pred_class = -1

        original_num_edges = graph_data.num_edges
        return total_reward, raw_fidelity, sparsity_score, subgraph_pred_class, original_pred_class, subgraph_data.num_edges, original_num_edges, subgraph_data, edge_mask

    def explain(self, graph_data_to_explain):
        self.actor.eval()
        self.state_gnn.eval()
        graph_data_to_explain = graph_data_to_explain.to(self.device)

        if graph_data_to_explain.num_nodes == 0:
            return graph_data_to_explain.clone(), torch.empty(0, dtype=torch.bool, device=self.device)
        if graph_data_to_explain.edge_index is None or graph_data_to_explain.edge_index.size(1) == 0:
            return graph_data_to_explain.clone(), torch.empty(0, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            _, node_embeddings_for_explain = self.state_gnn(
                graph_data_to_explain.x,
                graph_data_to_explain.edge_index,
                torch.zeros(graph_data_to_explain.num_nodes,
                            dtype=torch.long, device=self.device)
            )
        agent_states = node_embeddings_for_explain
        node_action_logits = self.select_actions(agent_states, add_noise=False)

        # Use deterministic mask for explain (same as eval)
        explanation_edge_mask, edge_probs = self._actions_to_edge_mask(
            graph_data_to_explain, node_action_logits)  # actor is in eval mode
        explanation_subgraph_data = self._get_subgraph(
            graph_data_to_explain, explanation_edge_mask)

        if explanation_subgraph_data.num_edges == 0 and graph_data_to_explain.num_edges > 0:
            # Fallback like in evaluate_graph
            sorted_probs, sorted_indices = torch.sort(
                edge_probs, descending=True)
            num_edges_to_keep = min(
                max(1, int(graph_data_to_explain.num_edges * 0.1)), 10)

            fallback_mask = torch.zeros_like(edge_probs, dtype=torch.bool)
            if num_edges_to_keep > 0 and len(sorted_indices) >= num_edges_to_keep:
                fallback_mask[sorted_indices[:num_edges_to_keep]] = True
            explanation_subgraph_data = self._get_subgraph(
                graph_data_to_explain, fallback_mask)
            explanation_edge_mask = fallback_mask

        return explanation_subgraph_data, explanation_edge_mask, edge_probs

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = {
            'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
            'state_gnn': self.state_gnn.state_dict()
        }
        torch.save(model_state, path)

    def load_model(self, path):
        model_state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(model_state['actor'])
        self.critic.load_state_dict(model_state['critic'])
        self.state_gnn.load_state_dict(model_state['state_gnn'])
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.state_gnn.to(self.device)
        # self.target_actor.load_state_dict(self.actor.state_dict())
        # self.target_critic.load_state_dict(self.critic.state_dict())

