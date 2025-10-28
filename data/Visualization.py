from typing import List, Optional, Sequence, Tuple, Union, Dict

import torch
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

from graphxai.datasets import (
    Benzene,
    FluorideCarbonyl,
    BAHouse,
    BAWheel,
    AlkaneCarbonyl,
    Mutagenicity,
)
from torch_geometric.data import Data
from graphxai.utils.explanation import Explanation

# Utilities for conversions
from graphxai.utils.nx_conversion import to_networkx_conv

# Make project root importable first so that `from utils import ...` resolves to root `utils/`,
# not `data/utils.py` when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch
import os
from data.utils import get_dataset
from gnn.train_gnn_for_dataset import get_model
from utils import set_seed
from data.utils import filter_correct_data
from explainer.coa_explainer import CoAExplainer
from explainer.train_coaexplainer import train_coaexp
from graphxai.metrics.metrics_graph import graph_exp_acc_graph, graph_exp_faith_graph
import torch
from tqdm import tqdm
from graphxai.utils import Explanation
from graphxai.datasets import Benzene
from graphxai.explainers import SubgraphX
from results.utils import get_exp_method
from graphxai.utils.performance.load_exp import exp_exists_graph
import numpy as np
import argparse
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from metrics.edge_exp_metrics import (
    soft_mask_to_hard,
    fid_neg,
    fid_pos,
    jac_edge_max,
    jac_edge_all,
    faith_edge,
    sparsity_edge,
)


# ------------------------------
# Helpers for molecular labeling/coloring
# ------------------------------

_MUTAG_IDX_TO_ATOM = {
    0: "C",
    1: "O",
    2: "Cl",
    3: "H",
    4: "N",
    5: "F",
    6: "Br",
    7: "S",
    8: "P",
    9: "I",
    10: "Na",
    11: "K",
    12: "Li",
    13: "Ca",
}

# _ATOM_COLOR = {
# 	'C': "#78A2D8",
# 	'N': "#7979cb",
# 	'O': "#53dca8ff",
# 	'S': "#f2ce9e",
# 	'F': "#E3B997",
# 	'P': '#bdbdbd',
# 	'B': "#6866c2",
# 	'Cl': '#78c679',
# 	'Br': '#a6bddb',
# 	'I': '#756bb1',
# 	'Na': "#78bde2",
# 	'K': '#c7e9c0',
# 	'Li': '#fdd0a2',
# 	'Ca': '#fdae6b',
# 	'H': "#badff7ff",
# 	'*': '#969696',
# }

# _ATOM_COLOR = {
#     "C": "#90B4CF",
#     "N": "#8f96bd",
#     "O": "#A5B1DB",
#     "S": "#2EABC5",
#     "F": "#F5A37D",
#     "P": "#E88DB2",
#     "B": "#B15375",
#     "Cl": "#EFBE73",
#     "Br": "#CF94A7",
#     "I": "#C5B4D5",
#     "Na": "#B0A3D1",
#     "K": "#4FB1B2",
#     "Li": "#CF94A7",
#     "Ca": "#E3B997",
#     "H": "#87BFD6",
#     "*": "#F1EDE3",
# }

_ATOM_COLOR = {
    "C": "#4E79A7",  # cool blue
    "N": "#E15759",  # bold coral
    "O": "#76B7B2",  # teal
    "S": "#F28E2B",  # vivid orange
    "F": "#B07AA1",  # plum
    "P": "#59A14F",  # lush green
    "B": "#9C755F",  # warm brown
    "Cl": "#F7E17F",  # golden yellow
    "Br": "#BAB0AC",  # soft taupe
    "I": "#7F7F7F",  # neutral gray
    "Na": "#A0CBE8",  # breezy blue
    "K": "#FFBE7A",  # peach
    "Li": "#8CD17D",  # mint
    "Ca": "#C5B0D5",  # lavender
    "H": "#EDC948",  # light gold
    "*": "#C7C7C7",  # placeholder light gray
}
# Explicit ATOM_TYPES mapping for datasets with shared ordering
_DATASET_ATOM_TYPES = {
    "Benzene": [
        "C",
        "N",
        "O",
        "S",
        "F",
        "P",
        "Cl",
        "Br",
        "Na",
        "Ca",
        "I",
        "B",
        "H",
        "*",
    ],
    "FluorideCarbonyl": [
        "C",
        "N",
        "O",
        "S",
        "F",
        "P",
        "Cl",
        "Br",
        "Na",
        "Ca",
        "I",
        "B",
        "H",
        "*",
    ],
    "AlkaneCarbonyl": [
        "C",
        "N",
        "O",
        "S",
        "F",
        "P",
        "Cl",
        "Br",
        "Na",
        "Ca",
        "I",
        "B",
        "H",
        "*",
    ],
}

_HIGHLIGHT_COLOR = ["#F9373A", "#ff7f00", "#7a1fdb", "#f5e20f"]


def _infer_atom_index(x_row: torch.Tensor) -> Optional[int]:
    """Infer atom index from a node feature row.
    Supports: scalar index, one-hot vector, or multi-hot (take argmax).
    Returns None if cannot infer.
    """
    if x_row is None:
        return None
    if not torch.is_tensor(x_row):
        return None
    if x_row.dim() == 0:
        try:
            return int(x_row.item())
        except Exception:
            return None
    if x_row.dim() == 1 and x_row.numel() > 0:
        # Try argmax (works for one-hot or multi-hot best class)
        return int(torch.argmax(x_row).item())
    return None


def _atom_label_and_color(dataset_name: str, x_row: torch.Tensor) -> Tuple[str, str]:
    """Map node feature row to (label, color) for molecular datasets.
    Falls back to generic label and color when unknown.
    """
    idx = _infer_atom_index(x_row)
    label = "X"
    if dataset_name in {"Mutagenicity"}:
        if idx is not None:
            label = _MUTAG_IDX_TO_ATOM.get(idx, "X")
    else:
        # For Benzene/FluorideCarbonyl/AlkaneCarbonyl, use explicit ATOM_TYPES mapping
        if idx is not None and dataset_name in _DATASET_ATOM_TYPES:
            atoms = _DATASET_ATOM_TYPES[dataset_name]
            if 0 <= idx < len(atoms):
                label = atoms[idx]

    color = _ATOM_COLOR.get(label, "#87BFDC")
    return label, color


def _build_edge_maps(
    edge_index: torch.Tensor,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[int]]]:
    """Build mapping for directed and undirected edge lookups.

    Returns:
            dir_map: {(u,v): idx}
            undir_map: {(min(u,v), max(u,v)): [idxs...]}
    """
    dir_map: Dict[Tuple[int, int], int] = {}
    undir_map: Dict[Tuple[int, int], List[int]] = {}
    eidx = edge_index.t().tolist()
    for idx, (u, v) in enumerate(eidx):
        dir_map[(u, v)] = idx
        key = (u, v) if u <= v else (v, u)
        lst = undir_map.get(key)
        if lst is None:
            undir_map[key] = [idx]
        else:
            lst.append(idx)
    return dir_map, undir_map


def _edge_set_from_imp(
    exp: Explanation,
    threshold: float = 0.0,
    top_k: Optional[int] = None,
    top_ratio: Optional[float] = None,
) -> set:
    """根据 top_k / top_ratio / threshold 选择边索引集合。

    优先级: top_k > top_ratio > threshold
    - top_k: 取 edge_imp 值最大的 k 条（k 自动裁剪到合法范围）。
    - top_ratio: 取按比例 ceil(ratio * E) 条 (E=边数)，ratio ∈ (0,1]。
    - threshold: 取 edge_imp > threshold 的边。
    """
    edge_imp = exp.edge_imp
    E = edge_imp.numel()
    if E == 0:
        return set()
    if top_k is not None and top_k > 0:
        k = min(top_k, E)
        # topk 返回 indices
        vals, idxs = torch.topk(edge_imp, k, largest=True, sorted=False)
        return set(idxs.tolist())
    if top_ratio is not None and top_ratio > 0:
        r = min(1.0, float(top_ratio))
        k = max(1, int(math.ceil(r * E)))
        vals, idxs = torch.topk(edge_imp, k, largest=True, sorted=False)
        return set(idxs.tolist())
    # fallback threshold
    mask = (edge_imp > threshold).nonzero(as_tuple=True)[0].tolist()
    return set(mask)


def visualize_graph_explanation(
    data: Data,
    exp: Union[Explanation, Sequence[Explanation]],  # 可直接传单个 Explanation 或列表
    dataset_name: Optional[str] = None,
    highlight_color: Optional[str] = None,
    same_color_for_all_exps: bool = True,
    draw_both_directions: bool = True,
    edge_threshold: float = 0.0,
    top_k: Optional[int] = None,
    top_ratio: Optional[float] = None,
    node_size: int = 600,
    base_edge_color: str = "#7F7F7F",
    base_edge_width: float = 5.0,
    highlight_width: float = 9.0,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    show_edge_imp_values: bool = False,
    show_all_edge_imp_values: bool = False,
    edge_imp_agg: str = "sum",
    value_format: str = "{:.3f}",
    edge_label_position: str = "off",  # 'on' 或 'off'
    edge_label_offset: float = 0.02,  # off 时沿法向偏移比例
    debug_edge_labels: bool = False,  # 新增: 打印标签调试信息
):
    """绘制图和解释高亮。

    参数:
            data: PyG Data 对象，需包含 x, edge_index。
            exp: Explanation 或 Explanation 列表（真实解释可能是列表）。
            dataset_name: 数据集名称，用于分子图节点标签和配色（'Benzene','FluorideCarbonyl','AlkaneCarbonyl','Mutagenicity','BAHouse','BAWheel'）。
            highlight_color: （新增）同色模式下指定的高亮颜色；优先级高于全局 _HIGHLIGHT_COLOR；若为空则回退到全局或默认颜色。
            same_color_for_all_exps: 多个解释时是否用同一种颜色高亮所有边；为 False 则每个解释用不同颜色。
            draw_both_directions: edge_index 为双向时，True=两条边都画并分别高亮；False=合并为一条无向边（任一方向高亮则高亮）。
            edge_threshold: 边重要性阈值（>threshold 视为高亮）。
            top_k: （新增）优先级最高。选取每个解释 edge_imp 最大的 k 条边进行高亮；忽略 edge_threshold。
            top_ratio: （新增）当 top_k 为空时生效。按比例选取 ceil(ratio * E) 条边；忽略 edge_threshold。
            node_size, base_edge_color, base_edge_width, highlight_width: 绘制参数。
            colors: 多解释不同颜色模式下使用的颜色列表；默认使用 matplotlib.tab10。
                            也可通过全局变量 _HIGHLIGHT_COLOR 指定：
                              - 若为字符串，如 '#ff0000'，同色模式下使用它；
                              - 若为字符串列表，如 ['#ff0000','#00aa00', ...]，多色模式按解释序号轮换使用。
            show_edge_imp_values: 是否在“高亮的边”上显示 edge_imp 数值（与 show_all_edge_imp_values 互斥，后者优先）。
            show_all_edge_imp_values: 若为 True，对所有边（无论是否超过阈值高亮）显示聚合后的 edge_imp 数值。
            edge_imp_agg: 多解释时的聚合方式：'sum' | 'mean' | 'max' | 'list'。
            value_format: 数值格式化字符串（不适用于 'list' 内部每个值时仍会使用该格式）。
            edge_label_position: 'on'=标签在线上(使用 networkx 内置)，'off'=标签沿边中心法向偏移。
            edge_label_offset: 当 edge_label_position='off' 时，偏移幅度（相对于图坐标归一化尺度的系数）。
            标签统一为无边框、无背景纯文本（便于矢量图后期编辑）。
            ax: 可选的 Matplotlib 轴；未提供则创建。
            show: 是否 plt.show()。
            save_path: 如提供则保存图片到此路径。

    返回:
            matplotlib.axes.Axes: 本次绘制使用/创建的轴对象，可用于在外部组合成多子图 figure。

    兼容性说明:
            如果 Explanation.edge_imp 为布尔张量 (dtype=bool)，将自动转换为 float (True->1.0, False->0.0)，
            以支持 top_k / top_ratio / 数值聚合等操作。
    """
    assert isinstance(data, Data), "data 必须是 torch_geometric.data.Data"

    # --- 统一 Explanation 输入格式 ---
    if isinstance(exp, Explanation):
        exp_list: List[Explanation] = [exp]
    elif isinstance(exp, (list, tuple)):
        exp_list = list(exp)
    else:
        raise TypeError("exp 必须是 Explanation 或 (list/tuple of Explanation)")

    print(
        f"[Input] explanation_count={len(exp_list)} (单个或多个 Explanation 已标准化)"
    )
    for e in exp_list:
        assert isinstance(e, Explanation), "exp 内元素必须为 Explanation"
        if e.edge_imp is None:
            raise ValueError("Explanation.edge_imp 为空，无法高亮边")
        # 兼容 bool 类型 edge_imp: 转为 float
        if isinstance(e.edge_imp, torch.Tensor) and e.edge_imp.dtype == torch.bool:
            print("[Normalize] edge_imp dtype=bool -> float (True=1.0, False=0.0)")
            e.edge_imp = e.edge_imp.float()
        # 尽量确保与 data 对齐
        if hasattr(e, "graph") and isinstance(e.graph, Data):
            # 不强制，但可用于一致性检查
            pass

    # Console prints: x, edge_index, counts
    x_info = f"x.shape={tuple(data.x.shape) if hasattr(data, 'x') and data.x is not None else 'None'}"
    eidx = data.edge_index
    e_info = f"edge_index.shape={tuple(eidx.shape)}"
    num_nodes = (
        data.num_nodes
        if hasattr(data, "num_nodes") and data.num_nodes is not None
        else (int(torch.max(eidx)) + 1 if eidx.numel() > 0 else 0)
    )
    num_edges_directed = eidx.shape[1]
    # 去重后的无向边数
    _, undir_map = _build_edge_maps(eidx)
    num_edges_undirected = len(undir_map)

    print("[Graph]", x_info, e_info)
    print(
        f"[Graph] 节点数(推断)={num_nodes} | 边数(双向)={num_edges_directed} | 边数(无向去重)={num_edges_undirected}"
    )

    # Explanation edge counts
    per_exp_counts_directed: List[int] = []
    per_exp_counts_undirected_unique: List[int] = []
    for i, e in enumerate(exp_list):
        idxs = list(
            _edge_set_from_imp(
                e, threshold=edge_threshold, top_k=top_k, top_ratio=top_ratio
            )
        )
        per_exp_counts_directed.append(len(idxs))
        # Unique undirected edges explained by this exp
        uniq_keys = set()
        for j in idxs:
            u = int(eidx[0, j].item())
            v = int(eidx[1, j].item())
            key = (u, v) if u <= v else (v, u)
            uniq_keys.add(key)
        per_exp_counts_undirected_unique.append(len(uniq_keys))

    if len(exp_list) == 1:
        multi_mode = False
        print(
            f"[Exp] 解释边数(双向)={per_exp_counts_directed[0]} | 解释边数(无向唯一)={per_exp_counts_undirected_unique[0]}"
        )
    else:
        multi_mode = True
        total_directed = sum(per_exp_counts_directed)
        union_undirected = set()
        for e in exp_list:
            idxs = list(
                _edge_set_from_imp(
                    e, threshold=edge_threshold, top_k=top_k, top_ratio=top_ratio
                )
            )
            for j in idxs:
                u = int(eidx[0, j].item())
                v = int(eidx[1, j].item())
                key = (u, v) if u <= v else (v, u)
                union_undirected.add(key)

    # 显示选择模式
    if top_k is not None and top_k > 0:
        print(f"[SelectMode] top_k={top_k}")
    elif top_ratio is not None and top_ratio > 0:
        print(f"[SelectMode] top_ratio={top_ratio}")
    else:
        print(f"[SelectMode] threshold>{edge_threshold}")

    if multi_mode:
        print("[Exp] 每种解释边数(双向)=", per_exp_counts_directed)
        print("[Exp] 每种解释边数(无向唯一)=", per_exp_counts_undirected_unique)
        print(
            f"[Exp] 所有解释边数合计(双向求和)={total_directed} | 所有解释边数合计(无向唯一并集)={len(union_undirected)}"
        )

    # Build NetworkX graph for drawing
    to_undirected = not draw_both_directions
    G = to_networkx_conv(data, to_undirected=to_undirected, remove_self_loops=True)
    pos = nx.kamada_kawai_layout(G)

    # Determine node border colors from original palette; fill stays white.
    node_border_colors: List[str] = []
    if dataset_name in {
        "Benzene",
        "FluorideCarbonyl",
        "AlkaneCarbonyl",
        "Mutagenicity",
    }:
        for n in G.nodes():
            x_row = data.x[n] if hasattr(data, "x") and data.x is not None else None
            _, color = _atom_label_and_color(dataset_name or "", x_row)
            node_border_colors.append(color)
    else:
        node_border_colors = [_ATOM_COLOR["C"] for _ in G.nodes()]
    node_face_colors = ["#FFFFFF" for _ in node_border_colors]

    # Base plot
    create_ax = ax is None
    if create_ax:
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_face_colors,
        node_size=node_size,
        ax=ax,
        linewidths=2.0,
        edgecolors=node_border_colors,
    )
    # Reuse shared kwargs so directed edges visibly show arrowheads when requested.
    edge_draw_common_kwargs: Dict[str, object] = {}
    if draw_both_directions and G.is_directed():
        edge_draw_common_kwargs.update(
            {
                "arrows": True,
                "arrowstyle": "->",
                "arrowsize": 30,
                "min_source_margin": 6,
                "min_target_margin": 6,
            }
        )
    else:
        edge_draw_common_kwargs["arrows"] = False
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        edge_color=base_edge_color,
        width=base_edge_width,
        ax=ax,
        **edge_draw_common_kwargs,
    )

    # Prepare highlighting
    dir_map, undir_map = _build_edge_maps(eidx)
    if colors is None or len(colors) == 0:
        # 10 distinct colors from tab10
        colors = [plt.get_cmap("tab10")(i) for i in range(10)]

    # Resolve override colors from global _HIGHLIGHT_COLOR
    override_colors: Optional[List[str]] = None
    # 单独参数 highlight_color 最优先（仅同色模式）
    if highlight_color is not None and len(str(highlight_color)) > 0:
        override_colors = [highlight_color]
    else:
        if isinstance(_HIGHLIGHT_COLOR, str) and len(_HIGHLIGHT_COLOR) > 0:
            override_colors = [_HIGHLIGHT_COLOR]
        elif isinstance(_HIGHLIGHT_COLOR, (list, tuple)) and len(_HIGHLIGHT_COLOR) > 0:
            override_colors = list(_HIGHLIGHT_COLOR)

    def _edges_for_exp_single(e: Explanation) -> List[Tuple[int, int]]:
        idxs = _edge_set_from_imp(e, edge_threshold, top_k=top_k, top_ratio=top_ratio)
        if draw_both_directions:
            return [
                (u, v)
                for (u, v), idx in dir_map.items()
                if idx in idxs and (u in G and v in G and G.has_edge(u, v))
            ]
        else:
            edgelist = []
            for (a, b), lst in undir_map.items():
                if any((idx in idxs) for idx in lst):
                    if G.has_edge(a, b) or G.has_edge(b, a):
                        edgelist.append((a, b))
            return edgelist

    def _prepare_edge_labels(
        label_edges: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], str]:
        labels: Dict[Tuple[int, int], str] = {}
        for u, v in label_edges:
            vals = []
            if draw_both_directions:
                for e in exp_list:
                    idx = dir_map.get((u, v), None)
                    if idx is not None:
                        vals.append(float(e.edge_imp[idx].item()))
            else:
                key = (u, v) if u <= v else (v, u)
                idx_list = undir_map.get(key, [])
                if len(idx_list) == 0:
                    continue
                for e in exp_list:
                    vals_e = [float(e.edge_imp[i].item()) for i in idx_list]
                    vals.append(max(vals_e))
            if len(vals) == 0:
                continue
            if edge_imp_agg == "sum":
                val_show = sum(vals)
            elif edge_imp_agg == "mean":
                val_show = sum(vals) / len(vals)
            elif edge_imp_agg == "max":
                val_show = max(vals)
            elif edge_imp_agg == "list":
                val_show = "[" + ",".join(value_format.format(v) for v in vals) + "]"
            else:
                val_show = sum(vals)
            labels[(u, v)] = (
                value_format.format(val_show) if edge_imp_agg != "list" else val_show
            )
        return labels

    def _draw_edge_labels(labels: Dict[Tuple[int, int], str]):
        if not labels:
            if debug_edge_labels:
                print("[Debug][EdgeLabel] 无标签可绘制")
            return
        if debug_edge_labels:
            # 仅打印前 8 条避免过长
            items_preview = list(labels.items())[:8]
            print(
                f"[Debug][EdgeLabel] 待绘制标签数量={len(labels)} 示例={items_preview}"
            )
        if edge_label_position == "on":
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=labels, font_size=8, ax=ax, label_pos=0.5
            )
        else:
            # off-edge: manual placement with normal offset
            # 估计坐标尺度，用于将 offset 比例转为实际偏移
            x_vals = [p[0] for p in pos.values()]
            y_vals = [p[1] for p in pos.values()]
            x_range = (max(x_vals) - min(x_vals)) if len(x_vals) > 1 else 1.0
            y_range = (max(y_vals) - min(y_vals)) if len(y_vals) > 1 else 1.0
            scale = 0.5 * (x_range + y_range)
            off = edge_label_offset * scale
            for (u, v), text in labels.items():
                (x1, y1) = pos[u]
                (x2, y2) = pos[v]
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = (x2 - x1), (y2 - y1)
                length = math.sqrt(dx * dx + dy * dy) + 1e-12
                # 法向 ( -dy, dx )
                nx_norm, ny_norm = -dy / length, dx / length
                # 修复: 使用 ax.text 而非 plt.text，避免标签误画到其他子图
                ax.text(
                    mx + nx_norm * off,
                    my + ny_norm * off,
                    text,
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                )

    # Draw highlights
    edge_draw_highlight_kwargs = edge_draw_common_kwargs.copy()
    if edge_draw_highlight_kwargs.get("arrows", False):
        edge_draw_highlight_kwargs["arrowsize"] = (
            edge_draw_highlight_kwargs.get("arrowsize", 28) * 1.2
        )
    if len(exp_list) > 0:
        if same_color_for_all_exps:
            union_edges = set()
            for e in exp_list:
                for uv in _edges_for_exp_single(e):
                    union_edges.add(
                        uv if draw_both_directions or uv[0] <= uv[1] else (uv[1], uv[0])
                    )
            if union_edges:
                edge_col = (
                    override_colors[0]
                    if (override_colors is not None and len(override_colors) > 0)
                    else colors[0]
                )
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(union_edges),
                    edge_color=edge_col,
                    width=highlight_width,
                    ax=ax,
                    **edge_draw_highlight_kwargs,
                )
            # Edge labels (aggregated)
            if show_all_edge_imp_values or show_edge_imp_values:
                if show_all_edge_imp_values:
                    label_edges = list(G.edges())
                else:
                    label_edges = list(union_edges)
                labels = _prepare_edge_labels(label_edges)
                _draw_edge_labels(labels)
        else:
            for i, e in enumerate(exp_list[::-1]):
                edgelist = _edges_for_exp_single(e)
                if edgelist:
                    col = (
                        override_colors[i % len(override_colors)]
                        if (override_colors is not None and len(override_colors) > 0)
                        else colors[i % len(colors)]
                    )
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        edge_color=col,
                        width=highlight_width,
                        ax=ax,
                        **edge_draw_highlight_kwargs,
                    )
            # Edge labels for multi-color mode
            if show_all_edge_imp_values or show_edge_imp_values:
                if show_all_edge_imp_values:
                    label_edges = list(G.edges())
                else:
                    all_edges = set()
                    for e in exp_list:
                        for uv in _edges_for_exp_single(e):
                            all_edges.add(
                                uv
                                if draw_both_directions or uv[0] <= uv[1]
                                else (uv[1], uv[0])
                            )
                    label_edges = list(all_edges)
                labels = _prepare_edge_labels(label_edges)
                _draw_edge_labels(labels)

    ax.set_axis_off()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def visualize_graph_explanation_style2(
    data: Data,
    exp: Union[Explanation, Sequence[Explanation]],  # 可直接传单个 Explanation 或列表
    dataset_name: Optional[str] = None,
    highlight_color: Optional[str] = None,
    same_color_for_all_exps: bool = True,
    draw_both_directions: bool = True,
    edge_threshold: float = 0.0,
    top_k: Optional[int] = None,
    top_ratio: Optional[float] = None,
    node_size: int = 600,
    base_edge_color: str = "#7F7F7F",
    base_edge_width: float = 5.0,
    highlight_width: float = 9.0,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    show_edge_imp_values: bool = False,
    show_all_edge_imp_values: bool = False,
    edge_imp_agg: str = "sum",
    value_format: str = "{:.3f}",
    edge_label_position: str = "off",  # 'on' 或 'off'
    edge_label_offset: float = 0.02,  # off 时沿法向偏移比例
    debug_edge_labels: bool = False,  # 新增: 打印标签调试信息
):
    """绘制图和解释高亮。

    参数:
            data: PyG Data 对象，需包含 x, edge_index。
            exp: Explanation 或 Explanation 列表（真实解释可能是列表）。
            dataset_name: 数据集名称，用于分子图节点标签和配色（'Benzene','FluorideCarbonyl','AlkaneCarbonyl','Mutagenicity','BAHouse','BAWheel'）。
            highlight_color: （新增）同色模式下指定的高亮颜色；优先级高于全局 _HIGHLIGHT_COLOR；若为空则回退到全局或默认颜色。
            same_color_for_all_exps: 多个解释时是否用同一种颜色高亮所有边；为 False 则每个解释用不同颜色。
            draw_both_directions: edge_index 为双向时，True=两条边都画并分别高亮；False=合并为一条无向边（任一方向高亮则高亮）。
            edge_threshold: 边重要性阈值（>threshold 视为高亮）。
            top_k: （新增）优先级最高。选取每个解释 edge_imp 最大的 k 条边进行高亮；忽略 edge_threshold。
            top_ratio: （新增）当 top_k 为空时生效。按比例选取 ceil(ratio * E) 条边；忽略 edge_threshold。
            node_size, base_edge_color, base_edge_width, highlight_width: 绘制参数。
            colors: 多解释不同颜色模式下使用的颜色列表；默认使用 matplotlib.tab10。
                            也可通过全局变量 _HIGHLIGHT_COLOR 指定：
                              - 若为字符串，如 '#ff0000'，同色模式下使用它；
                              - 若为字符串列表，如 ['#ff0000','#00aa00', ...]，多色模式按解释序号轮换使用。
            show_edge_imp_values: 是否在“高亮的边”上显示 edge_imp 数值（与 show_all_edge_imp_values 互斥，后者优先）。
            show_all_edge_imp_values: 若为 True，对所有边（无论是否超过阈值高亮）显示聚合后的 edge_imp 数值。
            edge_imp_agg: 多解释时的聚合方式：'sum' | 'mean' | 'max' | 'list'。
            value_format: 数值格式化字符串（不适用于 'list' 内部每个值时仍会使用该格式）。
            edge_label_position: 'on'=标签在线上(使用 networkx 内置)，'off'=标签沿边中心法向偏移。
            edge_label_offset: 当 edge_label_position='off' 时，偏移幅度（相对于图坐标归一化尺度的系数）。
            标签统一为无边框、无背景纯文本（便于矢量图后期编辑）。
            ax: 可选的 Matplotlib 轴；未提供则创建。
            show: 是否 plt.show()。
            save_path: 如提供则保存图片到此路径。

    返回:
            matplotlib.axes.Axes: 本次绘制使用/创建的轴对象，可用于在外部组合成多子图 figure。

    兼容性说明:
            如果 Explanation.edge_imp 为布尔张量 (dtype=bool)，将自动转换为 float (True->1.0, False->0.0)，
            以支持 top_k / top_ratio / 数值聚合等操作。
    """
    assert isinstance(data, Data), "data 必须是 torch_geometric.data.Data"

    # --- 统一 Explanation 输入格式 ---
    if isinstance(exp, Explanation):
        exp_list: List[Explanation] = [exp]
    elif isinstance(exp, (list, tuple)):
        exp_list = list(exp)
    else:
        raise TypeError("exp 必须是 Explanation 或 (list/tuple of Explanation)")

    print(
        f"[Input] explanation_count={len(exp_list)} (单个或多个 Explanation 已标准化)"
    )
    for e in exp_list:
        assert isinstance(e, Explanation), "exp 内元素必须为 Explanation"
        if e.edge_imp is None:
            raise ValueError("Explanation.edge_imp 为空，无法高亮边")
        # 兼容 bool 类型 edge_imp: 转为 float
        if isinstance(e.edge_imp, torch.Tensor) and e.edge_imp.dtype == torch.bool:
            print("[Normalize] edge_imp dtype=bool -> float (True=1.0, False=0.0)")
            e.edge_imp = e.edge_imp.float()
        # 尽量确保与 data 对齐
        if hasattr(e, "graph") and isinstance(e.graph, Data):
            # 不强制，但可用于一致性检查
            pass

    # Console prints: x, edge_index, counts
    x_info = f"x.shape={tuple(data.x.shape) if hasattr(data, 'x') and data.x is not None else 'None'}"
    eidx = data.edge_index
    e_info = f"edge_index.shape={tuple(eidx.shape)}"
    num_nodes = (
        data.num_nodes
        if hasattr(data, "num_nodes") and data.num_nodes is not None
        else (int(torch.max(eidx)) + 1 if eidx.numel() > 0 else 0)
    )
    num_edges_directed = eidx.shape[1]
    # 去重后的无向边数
    _, undir_map = _build_edge_maps(eidx)
    num_edges_undirected = len(undir_map)

    print("[Graph]", x_info, e_info)
    print(
        f"[Graph] 节点数(推断)={num_nodes} | 边数(双向)={num_edges_directed} | 边数(无向去重)={num_edges_undirected}"
    )

    # Explanation edge counts
    per_exp_counts_directed: List[int] = []
    per_exp_counts_undirected_unique: List[int] = []
    for i, e in enumerate(exp_list):
        idxs = list(
            _edge_set_from_imp(
                e, threshold=edge_threshold, top_k=top_k, top_ratio=top_ratio
            )
        )
        per_exp_counts_directed.append(len(idxs))
        # Unique undirected edges explained by this exp
        uniq_keys = set()
        for j in idxs:
            u = int(eidx[0, j].item())
            v = int(eidx[1, j].item())
            key = (u, v) if u <= v else (v, u)
            uniq_keys.add(key)
        per_exp_counts_undirected_unique.append(len(uniq_keys))

    if len(exp_list) == 1:
        multi_mode = False
        print(
            f"[Exp] 解释边数(双向)={per_exp_counts_directed[0]} | 解释边数(无向唯一)={per_exp_counts_undirected_unique[0]}"
        )
    else:
        multi_mode = True
        total_directed = sum(per_exp_counts_directed)
        union_undirected = set()
        for e in exp_list:
            idxs = list(
                _edge_set_from_imp(
                    e, threshold=edge_threshold, top_k=top_k, top_ratio=top_ratio
                )
            )
            for j in idxs:
                u = int(eidx[0, j].item())
                v = int(eidx[1, j].item())
                key = (u, v) if u <= v else (v, u)
                union_undirected.add(key)

    # 显示选择模式
    if top_k is not None and top_k > 0:
        print(f"[SelectMode] top_k={top_k}")
    elif top_ratio is not None and top_ratio > 0:
        print(f"[SelectMode] top_ratio={top_ratio}")
    else:
        print(f"[SelectMode] threshold>{edge_threshold}")

    if multi_mode:
        print("[Exp] 每种解释边数(双向)=", per_exp_counts_directed)
        print("[Exp] 每种解释边数(无向唯一)=", per_exp_counts_undirected_unique)
        print(
            f"[Exp] 所有解释边数合计(双向求和)={total_directed} | 所有解释边数合计(无向唯一并集)={len(union_undirected)}"
        )

    # Build NetworkX graph for drawing
    to_undirected = not draw_both_directions
    G = to_networkx_conv(data, to_undirected=to_undirected, remove_self_loops=True)
    pos = nx.kamada_kawai_layout(G)

    show_node_labels = dataset_name in {
        "Benzene",
        "FluorideCarbonyl",
        "AlkaneCarbonyl",
        "Mutagenicity",
    }
    node_border_colors: List[str] = []
    if show_node_labels:
        for n in G.nodes():
            x_row = data.x[n] if hasattr(data, "x") and data.x is not None else None
            label, color = _atom_label_and_color(dataset_name or "", x_row)
            G.nodes[n]["_label"] = label
            node_border_colors.append(color)
    else:
        node_border_colors = [_ATOM_COLOR["C"] for _ in G.nodes()]
    node_face_colors = ["#FFFFFF" for _ in node_border_colors]

    # Base plot
    create_ax = ax is None
    if create_ax:
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_face_colors,
        node_size=node_size,
        ax=ax,
        linewidths=2.0,
        edgecolors=node_border_colors,
    )
    # Reuse shared kwargs so directed edges visibly show arrowheads when requested.
    edge_draw_common_kwargs: Dict[str, object] = {}
    if draw_both_directions and G.is_directed():
        edge_draw_common_kwargs.update(
            {
                "arrows": True,
                "arrowstyle": "->",
                "arrowsize": 30,
                "min_source_margin": 6,
                "min_target_margin": 6,
            }
        )
    else:
        edge_draw_common_kwargs["arrows"] = False
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        edge_color=base_edge_color,
        width=base_edge_width,
        ax=ax,
        **edge_draw_common_kwargs,
    )
    if show_node_labels:
        labels = {n: G.nodes[n].get("_label", str(n)) for n in G.nodes()}
        for node_id, (x_coord, y_coord) in pos.items():
            ax.text(
                x_coord,
                y_coord,
                labels[node_id],
                fontsize=24,
                ha="center",
                va="center",
                color="#1A1A1A",
                fontweight="semibold",
            )

    # Prepare highlighting
    dir_map, undir_map = _build_edge_maps(eidx)
    if colors is None or len(colors) == 0:
        # 10 distinct colors from tab10
        colors = [plt.get_cmap("tab10")(i) for i in range(10)]

    # Resolve override colors from global _HIGHLIGHT_COLOR
    override_colors: Optional[List[str]] = None
    # 单独参数 highlight_color 最优先（仅同色模式）
    if highlight_color is not None and len(str(highlight_color)) > 0:
        override_colors = [highlight_color]
    else:
        if isinstance(_HIGHLIGHT_COLOR, str) and len(_HIGHLIGHT_COLOR) > 0:
            override_colors = [_HIGHLIGHT_COLOR]
        elif isinstance(_HIGHLIGHT_COLOR, (list, tuple)) and len(_HIGHLIGHT_COLOR) > 0:
            override_colors = list(_HIGHLIGHT_COLOR)

    def _edges_for_exp_single(e: Explanation) -> List[Tuple[int, int]]:
        idxs = _edge_set_from_imp(e, edge_threshold, top_k=top_k, top_ratio=top_ratio)
        if draw_both_directions:
            return [
                (u, v)
                for (u, v), idx in dir_map.items()
                if idx in idxs and (u in G and v in G and G.has_edge(u, v))
            ]
        else:
            edgelist = []
            for (a, b), lst in undir_map.items():
                if any((idx in idxs) for idx in lst):
                    if G.has_edge(a, b) or G.has_edge(b, a):
                        edgelist.append((a, b))
            return edgelist

    def _prepare_edge_labels(
        label_edges: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], str]:
        labels: Dict[Tuple[int, int], str] = {}
        for u, v in label_edges:
            vals = []
            if draw_both_directions:
                for e in exp_list:
                    idx = dir_map.get((u, v), None)
                    if idx is not None:
                        vals.append(float(e.edge_imp[idx].item()))
            else:
                key = (u, v) if u <= v else (v, u)
                idx_list = undir_map.get(key, [])
                if len(idx_list) == 0:
                    continue
                for e in exp_list:
                    vals_e = [float(e.edge_imp[i].item()) for i in idx_list]
                    vals.append(max(vals_e))
            if len(vals) == 0:
                continue
            if edge_imp_agg == "sum":
                val_show = sum(vals)
            elif edge_imp_agg == "mean":
                val_show = sum(vals) / len(vals)
            elif edge_imp_agg == "max":
                val_show = max(vals)
            elif edge_imp_agg == "list":
                val_show = "[" + ",".join(value_format.format(v) for v in vals) + "]"
            else:
                val_show = sum(vals)
            labels[(u, v)] = (
                value_format.format(val_show) if edge_imp_agg != "list" else val_show
            )
        return labels

    def _draw_edge_labels(labels: Dict[Tuple[int, int], str]):
        if not labels:
            if debug_edge_labels:
                print("[Debug][EdgeLabel] 无标签可绘制")
            return
        if debug_edge_labels:
            # 仅打印前 8 条避免过长
            items_preview = list(labels.items())[:8]
            print(
                f"[Debug][EdgeLabel] 待绘制标签数量={len(labels)} 示例={items_preview}"
            )
        if edge_label_position == "on":
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=labels, font_size=8, ax=ax, label_pos=0.5
            )
        else:
            # off-edge: manual placement with normal offset
            # 估计坐标尺度，用于将 offset 比例转为实际偏移
            x_vals = [p[0] for p in pos.values()]
            y_vals = [p[1] for p in pos.values()]
            x_range = (max(x_vals) - min(x_vals)) if len(x_vals) > 1 else 1.0
            y_range = (max(y_vals) - min(y_vals)) if len(y_vals) > 1 else 1.0
            scale = 0.5 * (x_range + y_range)
            off = edge_label_offset * scale
            for (u, v), text in labels.items():
                (x1, y1) = pos[u]
                (x2, y2) = pos[v]
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = (x2 - x1), (y2 - y1)
                length = math.sqrt(dx * dx + dy * dy) + 1e-12
                # 法向 ( -dy, dx )
                nx_norm, ny_norm = -dy / length, dx / length
                # 修复: 使用 ax.text 而非 plt.text，避免标签误画到其他子图
                ax.text(
                    mx + nx_norm * off,
                    my + ny_norm * off,
                    text,
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                )

    # Draw highlights
    edge_draw_highlight_kwargs = edge_draw_common_kwargs.copy()
    if edge_draw_highlight_kwargs.get("arrows", False):
        edge_draw_highlight_kwargs["arrowsize"] = (
            edge_draw_highlight_kwargs.get("arrowsize", 28) * 1.2
        )
    if len(exp_list) > 0:
        if same_color_for_all_exps:
            union_edges = set()
            for e in exp_list:
                for uv in _edges_for_exp_single(e):
                    union_edges.add(
                        uv if draw_both_directions or uv[0] <= uv[1] else (uv[1], uv[0])
                    )
            if union_edges:
                edge_col = (
                    override_colors[0]
                    if (override_colors is not None and len(override_colors) > 0)
                    else colors[0]
                )
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(union_edges),
                    edge_color=edge_col,
                    width=highlight_width,
                    ax=ax,
                    **edge_draw_highlight_kwargs,
                )
            # Edge labels (aggregated)
            if show_all_edge_imp_values or show_edge_imp_values:
                if show_all_edge_imp_values:
                    label_edges = list(G.edges())
                else:
                    label_edges = list(union_edges)
                labels = _prepare_edge_labels(label_edges)
                _draw_edge_labels(labels)
        else:
            for i, e in enumerate(exp_list[::-1]):
                edgelist = _edges_for_exp_single(e)
                if edgelist:
                    col = (
                        override_colors[i % len(override_colors)]
                        if (override_colors is not None and len(override_colors) > 0)
                        else colors[i % len(colors)]
                    )
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        edge_color=col,
                        width=highlight_width,
                        ax=ax,
                        **edge_draw_highlight_kwargs,
                    )
            # Edge labels for multi-color mode
            if show_all_edge_imp_values or show_edge_imp_values:
                if show_all_edge_imp_values:
                    label_edges = list(G.edges())
                else:
                    all_edges = set()
                    for e in exp_list:
                        for uv in _edges_for_exp_single(e):
                            all_edges.add(
                                uv
                                if draw_both_directions or uv[0] <= uv[1]
                                else (uv[1], uv[0])
                            )
                    label_edges = list(all_edges)
                labels = _prepare_edge_labels(label_edges)
                _draw_edge_labels(labels)

    ax.set_axis_off()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# if __name__ == "__main__":
