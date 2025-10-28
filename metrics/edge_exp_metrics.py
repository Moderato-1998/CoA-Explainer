import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def soft_mask_to_hard(mask, type='topk', value=10):
    # handle empty mask
    if mask is None:
        return None
    if mask.numel() == 0:
        return mask.float()

    n = mask.numel()

    if type == 'topk':
        k = int(value)
        # clamp k to [0, n]
        if k <= 0:
            return torch.zeros_like(mask).float()
        if k >= n:
            return torch.ones_like(mask).float()
        # get topk indices and build hard mask
        _, indices = torch.topk(mask, k=k)
        hard_mask = torch.zeros_like(mask, dtype=torch.bool)
        hard_mask[indices] = True

    elif type == 'threshold':
        hard_mask = (mask >= value)

    elif type == 'ratio':
        # value is a ratio in (0,1)
        k = int(value * n)
        if k <= 0:
            return torch.zeros_like(mask).float()
        if k >= n:
            return torch.ones_like(mask).float()
        _, indices = torch.topk(mask, k=k)
        hard_mask = torch.zeros_like(mask, dtype=torch.bool)
        hard_mask[indices] = True

    else:
        # unknown type: fallback to threshold with given value
        hard_mask = (mask >= value)

    return hard_mask.float()

def _build_edge_only_subgraph(x_all: torch.Tensor, edge_index_all: torch.Tensor, edge_indices: torch.Tensor):
    """Given a set of edge indices (columns of edge_index_all), build a subgraph that
    contains only these edges and the nodes incident to them. Node indices are remapped
    to [0, num_sub_nodes). Returns (x_sub, edge_index_sub, batch_sub). If no edges,
    returns (None, None, None).
    """
    if edge_indices is None or edge_indices.numel() == 0:
        return None, None, None
    ei = edge_index_all[:, edge_indices]
    if ei.numel() == 0:
        return None, None, None
    nodes = torch.unique(ei)
    remap = torch.full((x_all.size(0),), -1, dtype=torch.long, device=x_all.device)
    remap[nodes] = torch.arange(nodes.size(0), device=x_all.device)
    ei_remap = remap[ei]
    x_sub = x_all[nodes]
    b_sub = torch.zeros(x_sub.size(0), dtype=torch.long, device=x_all.device)
    return x_sub, ei_remap, b_sub

def fid_neg(data, exp, model, pred_class):
    # 仅使用解释边（edge_imp==1）的边构建子图进行预测
    edge_indices = torch.where(exp.edge_imp == 1.0)[0]
    x_sub, e_idx_sub, b_sub = _build_edge_only_subgraph(data.x, data.edge_index, edge_indices)
    if e_idx_sub is not None:
        subgraph_out, _ = model(x_sub, e_idx_sub, batch=b_sub)
        subgraph_pred_class = subgraph_out.argmax(dim=1).item()
        return 1 - (subgraph_pred_class == pred_class)
    else:
        # 无边/无节点时视为失败
        return 1

def fid_pos(data, exp, model, pred_class):
    # 仅使用补图边（edge_imp==0）的边构建子图进行预测
    edge_indices = torch.where(exp.edge_imp == 0.0)[0]
    x_sub, e_idx_sub, b_sub = _build_edge_only_subgraph(data.x, data.edge_index, edge_indices)
    if e_idx_sub is not None:
        pos_subgraph_out, _ = model(x_sub, e_idx_sub, batch=b_sub)
        pos_subgraph_pred_class = pos_subgraph_out.argmax(dim=1).item()
        return 1 - (pos_subgraph_pred_class == pred_class)
    else:
        return 1
    
def jac_edge_max(gt_exp, generated_exp):
    '''
    Specifically for graph-level explanation accuracy

    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    EPS = 1e-09
    JAC_edge = None

    if generated_exp.edge_imp is not None:
        JAC_edge = []
        for exp in gt_exp:
            TPs = []
            FPs = []
            FNs = []
            true_edges = torch.where(exp.edge_imp == 1.0)[0]
            for edge in range(exp.edge_imp.shape[0]):
                if generated_exp.edge_imp[edge]:
                    if edge in true_edges:
                        TPs.append(edge)
                    else:
                        FPs.append(edge)
                else:
                    if edge in true_edges:
                        FNs.append(edge)
            TP = len(TPs)
            FP = len(FPs)
            FN = len(FNs)
            JAC_edge.append(TP / (TP + FP + FN + EPS))

        JAC_edge = max(JAC_edge)

    return JAC_edge

def jac_edge_all(gt_exp, generated_exp):

    EPS = 1e-09
    all_true_edges = set()
    for exp in gt_exp:
        true_edges = torch.where(exp.edge_imp == 1.0)[0]
        for edge in true_edges:
            all_true_edges.add(edge.item())
    
    TPs = []
    FPs = []
    FNs = []
    for edge in range(exp.edge_imp.shape[0]):
        if generated_exp.edge_imp[edge]:
            if edge in true_edges:
                TPs.append(edge)
            else:
                FPs.append(edge)
        else:
            if edge in true_edges:
                FNs.append(edge)
    TP = len(TPs)
    FP = len(FPs)
    FN = len(FNs)
    JAC_edge = TP / (TP + FP + FN + EPS)

    return JAC_edge

def faith_edge(generated_exp, data, model, forward_kwargs = {}):
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
        forward_kwargs (dict, optional): Any additional arguments to the forward method
            of model, other than x and edge_index.
    '''

    GEF_edge = None

    X = data.x.to(device)
    EIDX = data.edge_index.to(device)

    # 原始整图预测
    org_vec, _ = model(X, EIDX, **forward_kwargs)
    org_softmax = F.softmax(org_vec, dim=-1)

    if generated_exp.edge_imp is not None:
        # 仅保留解释边，构造“边驱动”的子图（不包含同节点间未激活边），并重映射节点
        keep_edges = torch.where(generated_exp.edge_imp == 1)[0]
        x_sub, e_idx_sub, b_sub = _build_edge_only_subgraph(X, EIDX, keep_edges)

        if e_idx_sub is None:
            # 无边/无节点：与 fid 系列保持一致，视作失败
            return 1

        # 使用子图及其 batch 进行前向，避免孤立节点影响
        kw = dict(forward_kwargs) if isinstance(forward_kwargs, dict) else {}
        kw['batch'] = b_sub
        pert_vec, _ = model(x_sub, e_idx_sub.to(device), **kw)
        pert_softmax = F.softmax(pert_vec, dim=-1)
        
        # GEF_edge = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()
        # Add epsilon to avoid log(0) which causes nan
        eps = 1e-8
        org_softmax_safe = torch.clamp(org_softmax, min=eps)
        pert_softmax_safe = torch.clamp(pert_softmax, min=eps)
        GEF_edge = 1 - torch.exp(-F.kl_div(org_softmax_safe.log(), pert_softmax_safe, None, None, 'sum')).item()
        # if GEF_edge > 0.999:
        #     GEF_edge = 1.0
        
    return GEF_edge

def sparsity_edge(generated_exp):
    '''
    Args:
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    GES_edge = None

    if generated_exp.edge_imp is not None:
        GES_edge = 1 - (generated_exp.edge_imp.sum() / generated_exp.edge_imp.numel()).item()

    return GES_edge

def complete_match_cont(gt_exp, generated_exp):
    # 统计找出完整解释的数量（多了少了都不算完整）
    # 需要考虑有些图可能有多个正确解释
    pass