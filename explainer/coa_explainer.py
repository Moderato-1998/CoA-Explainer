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
        input_dim = (
            global_state_dim + global_action_dim
            if global_action_dim > 0
            else global_state_dim
        )
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
    def __init__(
        self,
        gnn_model,
        state_encode,
        num_node_features,
        max_nodes,
        node_embedding_dim,
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.01,
        device="cpu",
        batch_size=64,
        buffer_capacity=10000,
        exploration_noise_std: float = 0.05,
        # Continuous relaxation temperature for edge mask (Gumbel-Sigmoid)
        gumbel_temperature: float = 1.0,
        steps_per_episode=3,
    ):
        # This code will be made publicly available upon the acceptance of the paper.
