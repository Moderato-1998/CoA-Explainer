import torch
from torch import nn
from typing import Optional, Tuple, Union

try:
    from torch_geometric.nn import global_mean_pool
except Exception:  # Fallback if torch_geometric is not available at import time
    global_mean_pool = None  # type: ignore


class RCModelAdapter(nn.Module):
    """
    A lightweight adapter to make existing GNN models (that return (graph_logits, node_emb)
    in forward) compatible with RC_Explainer.

    It provides:
    - forward(x, edge_index, edge_attr=None, batch=None) -> graph logits (Tensor)
    - get_node_reps(x, edge_index, edge_attr=None, batch=None) -> node embeddings (Tensor)
    - get_graph_rep(x, edge_index, edge_attr=None, batch=None) -> graph embedding BEFORE final classifier (Tensor)

    Assumptions:
    - Base model's forward returns a tuple: (graph_logits, node_emb)
    - Graph embedding used by RC should be the pooled node_emb with shape [batch_size, hidden_channels]
    - edge_attr is accepted but ignored (kept for API compatibility)
    """

    def __init__(self, base_model: nn.Module, hidden_channels_attr: str = 'hidden_channels') -> None:
        super().__init__()
        self.base_model = base_model
        # Expose hidden size if available for convenience
        self.hidden_channels: Optional[int] = getattr(base_model, hidden_channels_attr, None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return graph-level logits only, matching rc_explainer's expectations."""
        out = self.base_model(x, edge_index, batch)
        if isinstance(out, tuple):
            g_emb = out[0]
        else:
            g_emb = out
        return g_emb

    def get_node_reps(self, x: torch.Tensor, edge_index: torch.Tensor,
                       edge_attr: Optional[torch.Tensor] = None,
                       batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.base_model(x, edge_index, batch)
        if isinstance(out, tuple) and len(out) > 1:
            n_emb = out[1]
            return n_emb
        raise ValueError("Base model must return (graph_logits, node_emb) to extract node reps.")

    def get_graph_rep(self, x: torch.Tensor, edge_index: torch.Tensor,
                      edge_attr: Optional[torch.Tensor] = None,
                      batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return graph representation with dimension hidden_channels via pooling node reps.

        This avoids using the classifier output dimension (out_channels), which wouldn't
        match RC_Explainer's expected hidden size.
        """
        out = self.base_model(x, edge_index, batch)
        if isinstance(out, tuple) and len(out) > 1:
            _, n_emb = out
            if global_mean_pool is None:
                raise RuntimeError("torch_geometric.global_mean_pool is required for get_graph_rep.")
            if batch is None:
                # If batch is None, treat the whole graph as a single graph (batch of zeros)
                batch = torch.zeros((n_emb.size(0),), dtype=torch.long, device=n_emb.device)
            g_rep = global_mean_pool(n_emb, batch)
            return g_rep
        # Fallback: if base model doesn't provide node embeddings, use its output as graph rep
        if isinstance(out, torch.Tensor):
            return out
        raise ValueError("Unable to derive graph representation from the base model output.")
