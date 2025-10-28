from torch_geometric.nn import GINConv, GINEConv, GATConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv


class GCN_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_2layer, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        n_emb = x.relu()

        g_emb = global_mean_pool(n_emb, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        g_emb = self.lin(g_emb)

        return g_emb, n_emb


class GCN_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_3layer, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=1, edge_weights=None):
        if edge_weights is None:

            x = self.conv1(x, edge_index)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            n_emb = self.conv3(x, edge_index)
            # n_emb = torch.nn.functional.normalize(n_emb, p=2, dim=1)
            # n_emb = n_emb.relu()
            # n_emb = torch.nn.functional.normalize(n_emb, p=2, dim=1)
        else:

            x = self.conv1(x, edge_index, edge_weight=edge_weights)

            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weight=edge_weights)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            n_emb = self.conv3(x, edge_index, edge_weight=edge_weights)
            # n_emb = torch.nn.functional.normalize(n_emb, p=2, dim=1)
            # n_emb = n_emb.relu()
            # n_emb = torch.nn.functional.normalize(n_emb, p=2, dim=1)
        g_emb = global_mean_pool(n_emb, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        g_emb = self.lin(g_emb)

        return g_emb, n_emb

    def embedding(self, x, edge_index, batch=1, edge_weights=None):
        if edge_weights is None:

            x = self.conv1(x, edge_index)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv3(x, edge_index)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            # x = x.relu()
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
        else:

            x = self.conv1(x, edge_index, edge_weight=edge_weights)

            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weight=edge_weights)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_weight=edge_weights)
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
            # x = x.relu()
            # x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = global_mean_pool(x, batch)

        # x = F.dropout(x, p=0.5, training=self.training)

        return x


class GINWConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weights: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weights, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:

        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GIN_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN_2layer, self).__init__()
        self.hidden_channels = hidden_channels
        self.mlp1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GINConv(self.mlp2)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        n_emb = x.relu()

        g_emb = global_mean_pool(n_emb, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        g_emb = self.lin(g_emb)

        return g_emb, n_emb


class GIN_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN_3layer, self).__init__()
        self.hidden_channels = hidden_channels
        self.mlp1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GINConv(self.mlp2)
        self.mlp3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = GINConv(self.mlp3)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        n_emb = x.relu()

        g_emb = global_mean_pool(n_emb, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        g_emb = self.lin(g_emb)

        return g_emb, n_emb


class GINW_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINW_3layer, self).__init__()
        self.mlp1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1 = GINWConv(self.mlp1, edge_dim=1)
        self.mlp2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GINWConv(self.mlp2, edge_dim=1)
        self.mlp3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = GINWConv(self.mlp3, edge_dim=1)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_weights=None):
        if edge_weights is None:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            x = x.relu()
        else:
            x = self.conv1(x, edge_index, edge_weights)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weights)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_weights)
            x = x.relu()

        x = global_mean_pool(x, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GAT_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_3layer, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        emb_n = x.relu()

        emb_g = global_mean_pool(emb_n, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        emb_g = self.lin(emb_g)

        return emb_g, emb_n

class GAT_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_2layer, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        emb_n = x.relu()

        emb_g = global_mean_pool(emb_n, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        emb_g = self.lin(emb_g)

        return emb_g, emb_n