import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import matplotlib.patheffects as pe

# workspace root
ROOT = r'd:\OneDrive\1-Code\GNN_Exp\my model\Multi_Agent_Reinforcement_Learning_Explainer\25-0920'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.utils import get_dataset, filter_correct_data
from gnn.train_gnn_for_dataset import get_model
from graphxai.utils.performance.load_exp import exp_exists_graph
from graphxai.utils import Explanation
from graphxai.metrics.metrics_graph import graph_exp_acc_graph


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config via env
DATASET = os.environ.get('FIG1_DATASET', 'Benzene')
MODEL = os.environ.get('FIG1_MODEL', 'GCN_3layer')
GNNE_NAME = os.environ.get('FIG1_BASELINE', 'gnnex')  # GNNExplainer 缓存目录名
OUR_NAME = os.environ.get('FIG1_OUR', 'coex')         # 我们方法缓存目录名
RATIO = float(os.environ.get('SAMPLE_RATIO', '0.2'))  # GNNExplainer 二值化比例
SORT_BY = os.environ.get('SAMPLE_SORT', 'gt')         # 'gt' 或 'idx'
MAX_SAMPLES = int(os.environ.get('SAMPLE_MAX', '2000'))
SKIP_EMPTY_GT = os.environ.get('SAMPLE_SKIP_EMPTY_GT', '1') not in ('0', 'false', 'False')
PLOT_MODE = os.environ.get('SAMPLE_PLOT_MODE', 'ribbon')  # 'ribbon' | 'bins' | 'line' | 'delta'
SMOOTH_WIN = int(os.environ.get('SAMPLE_SMOOTH_WIN', '11'))
BIN_NUM = int(os.environ.get('SAMPLE_BIN_NUM', '10'))
SHADE_ALPHA = float(os.environ.get('SAMPLE_SHADE_ALPHA', '0.25'))
LAYOUT = os.environ.get('SAMPLE_LAYOUT', 'kamada').strip().lower()  # 'kamada' | 'spring' | 'circular' | 'spectral'
# Control which eligible samples to display on the right (1 = lowest/highest GT sparsity)
LOW_RANK = max(1, int(os.environ.get('SAMPLE_LOW_RANK', '1')))
HIGH_RANK = max(1, int(os.environ.get('SAMPLE_HIGH_RANK', '1')))
# Seed to control small-panel layout randomness; for 'kamada', we use a seeded spring init
_SEED_ENV = os.environ.get('SAMPLE_LAYOUT_SEED', '1001').strip()
try:
    LAYOUT_SEED = int(_SEED_ENV)
except Exception:
    # allow strings like 'none' to mean no fixed seed
    LAYOUT_SEED = None

# Atom types and visualization controls
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']
ATOM_COLORS = {
    'C': '#aec7e8', # light blue
    'N': '#ffdd57',  # yellow
    'O': '#ff7f0e',  # orange
    'S': '#1f77b4',  # blue
    'F': '#2ca02c',  # green
    'P':  '#d62728',  # red
    'Cl': '#17becf', # cyan
    'Br': '#8c564b', # brown
    'Na': '#808080',  # gray
    'Ca': '#98df8a', # light green
    'I': '#9467bd',  # purple
    'B': '#bcbd22',  # olive
    'H': '#ffffff',  # white
    '*': '#bfbfbf',  # light gray
}
SHOW_NODE_LABELS = os.environ.get('SAMPLE_NODE_LABELS', '1') not in ('0', 'false', 'False')
COLORED_NODES = os.environ.get('SAMPLE_NODE_COLOR', '1') not in ('0', 'false', 'False')
NODE_FONT_SIZE = int(os.environ.get('SAMPLE_NODE_FONT_SIZE', '8'))
NODE_ALPHA = float(os.environ.get('SAMPLE_NODE_ALPHA', '0.9'))
LABEL_STROKE = os.environ.get('SAMPLE_NODE_LABEL_STROKE', '0') not in ('0', 'false', 'False')

# Optional mapping from atomic number to symbol (for datasets that store data.z)
ATOMIC_NUM_TO_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 15: 'P', 16: 'S', 17: 'Cl', 20: 'Ca', 35: 'Br', 53: 'I'
}

def infer_atom_labels_and_colors(data):
    n = int(data.num_nodes)
    labels = ['C'] * n  # default
    # Priority 1: integer-coded atom attribute (e.g., data.atom or data.z atomic number)
    tensor_candidates = []
    for name in ('atom', 'atoms', 'atom_type', 'atom_types', 'z'):
        if hasattr(data, name):
            t = getattr(data, name)
            if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.size(0) == n:
                tensor_candidates.append((name, t))
    if tensor_candidates:
        name, t = tensor_candidates[0]
        t = t.detach().cpu()
        if not torch.is_floating_point(t):
            arr = t.long().numpy()
        else:
            arr = t.round().long().numpy()
        # If this looks like atomic numbers (>= 1 and not too large), map via table
        if name == 'z' or (arr.min() >= 1 and arr.max() > len(ATOM_TYPES)):
            for i in range(n):
                sym = ATOMIC_NUM_TO_SYMBOL.get(int(arr[i]), '*')
                labels[i] = sym if sym in ATOM_COLORS else '*'
        else:
            # Treat as indices into ATOM_TYPES
            for i in range(n):
                idx = int(arr[i])
                labels[i] = ATOM_TYPES[idx] if 0 <= idx < len(ATOM_TYPES) else '*'
    elif hasattr(data, 'x') and isinstance(data.x, torch.Tensor) and data.x.dim() == 2 and data.x.size(0) == n:
        X = data.x.detach().cpu()
        C = X.size(1)
        # Assume first K dims correspond to ATOM_TYPES if possible
        K = min(C, len(ATOM_TYPES))
        sub = X[:, :K]
        idxs = torch.argmax(sub, dim=1).numpy()
        for i in range(n):
            labels[i] = ATOM_TYPES[int(idxs[i])]
    # Colors for each label
    # node_colors = [ATOM_COLORS.get(lab, '#bfbfbf') for lab in labels]
    node_colors = "#FFFFFF"
    return labels, node_colors

# Multiple ratios support for GNNE curves
_R_ENV = os.environ.get('SAMPLE_RATIOS', '').strip()
if _R_ENV:
    try:
        RATIOS = [float(x) for x in _R_ENV.split(',') if x.strip()]
    except Exception:
        RATIOS = [0.05, 0.1, 0.2, 0.4]
else:
    # 默认展示多条以形成对比
    RATIOS = [0.05, 0.1, 0.2, 0.4]
RATIOS = [min(1.0, max(1e-6, float(r))) for r in RATIOS]
RATIOS = sorted(set(RATIOS))

# paths
EXP_ROOT = os.path.join('results', 'exp_acc', DATASET, 'metrics_result', 'EXPS')
GNNE_LOC = os.path.join(EXP_ROOT, GNNE_NAME)
OUR_LOC = os.path.join(EXP_ROOT, OUR_NAME)
OUT_DIR = os.path.join('results', 'exp_acc', DATASET, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)


def top_ratio_mask(edge_imp: torch.Tensor, ratio: float) -> torch.Tensor:
    e = edge_imp
    if e is None or e.numel() == 0:
        return torch.zeros_like(e, dtype=torch.bool)
    r = float(ratio)
    if r <= 0:
        r = 1.0 / float(e.numel())
    if r > 1:
        r = 1.0
    k = int(math.ceil(r * e.numel()))
    k = max(1, min(k, e.numel()))
    top_idx = torch.topk(e, k=k).indices
    mask = torch.zeros_like(e, dtype=torch.bool)
    mask[top_idx] = True
    return mask


def main():
    # load dataset/model
    seed = 42
    dataset = get_dataset(DATASET, seed, device)
    model = get_model(dataset, MODEL, load_state=True)
    model.eval()

    # collect indices of correctly predicted test samples
    _, _, test_idx = filter_correct_data(dataset, model)

    # storage
    sample_ids = []
    s_gt, s_our = [], []
    # per-ratio storages for GNNE
    s_gn_map = {r: [] for r in RATIOS}
    gea_gn_map = {r: [] for r in RATIOS}
    gea_our = []

    # helper to merge Explanation or list[Explanation] to a vector
    def merge_edge_imp(exp_obj, E: int, prefer_continuous: bool = False):
        if exp_obj is None:
            return None
        # Explanation
        if hasattr(exp_obj, 'edge_imp'):
            ei = exp_obj.edge_imp
            if ei is None:
                return None
            return ei
        # list/tuple of Explanation
        if isinstance(exp_obj, (list, tuple)) and len(exp_obj) > 0:
            tensors = []
            for e in exp_obj:
                if hasattr(e, 'edge_imp') and e.edge_imp is not None:
                    tensors.append(e.edge_imp)
            if len(tensors) == 0:
                return None
            try:
                S = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
            except Exception:
                return None
            if prefer_continuous:
                # elementwise max as continuous importance aggregator
                return torch.max(S, dim=0).values
            else:
                # boolean union
                return (S > 0).any(dim=0).to(torch.float32)
        return None

    for idx in tqdm(test_idx, desc=f'Fig1-sample on {DATASET}: R={RATIO}'):
        data, gt_exp = dataset[idx]
        data = data.to(device)

        # ensure model predicts correctly
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index, batch=torch.zeros((1,), dtype=torch.long, device=device))
            pred = out.argmax(dim=1).item()
        if pred != data.y.item():
            continue

        # load explanations
        exp_gn = exp_exists_graph(idx, path=GNNE_LOC, get_exp=True)
        exp_our = exp_exists_graph(idx, path=OUR_LOC, get_exp=True)
        if exp_gn is None or exp_our is None:
            # 需要两者都存在，便于并排对比
            continue

        # number of edges in graph
        M = int(data.edge_index.size(1))
        if M == 0:
            continue


    # GT sparsity (union if multiple components)
        gt_imp = merge_edge_imp(gt_exp, M, prefer_continuous=False)
        if gt_imp is None or gt_imp.numel() != M:
            continue
        gt_mask = (gt_imp > 0).to(torch.bool).to(device)
        k_gt = int(gt_mask.sum().item())
        if SKIP_EMPTY_GT and k_gt == 0:
            # 跳过没有 GT 的样本（通常是 y=0）
            continue
        spars_gt = k_gt / float(M)

        # GNNExplainer: compute once continuous importance, then multiple ratios
        ei_gn = merge_edge_imp(exp_gn, M, prefer_continuous=True)
        if ei_gn is None or ei_gn.numel() != M:
            continue
        ei_gn = ei_gn.to(device)

        # OUR: assumed boolean mask saved
        ei_our = merge_edge_imp(exp_our, M, prefer_continuous=False)
        if ei_our is None or ei_our.numel() != M:
            continue
        mask_our = (ei_our > 0).to(torch.bool).to(device)
        k_our = int(mask_our.sum().item())
        spars_our = k_our / float(M)

        # GNNE per-ratio sparsity + GEA, and OUR GEA
        try:
            for r in RATIOS:
                mask_gn_r = top_ratio_mask(ei_gn, r)
                k_gn = int(mask_gn_r.sum().item())
                s_gn_map[r].append(k_gn / float(M))
                _, _, gea_val_gn = graph_exp_acc_graph(gt_exp, Explanation(edge_imp=mask_gn_r, graph=data))
                gea_gn_map[r].append(float(gea_val_gn))
            _, _, gea_val_our = graph_exp_acc_graph(gt_exp, Explanation(edge_imp=mask_our, graph=data))
        except Exception:
            continue

        sample_ids.append(int(idx))
        s_gt.append(spars_gt)
        s_our.append(spars_our)
        gea_our.append(float(gea_val_our))

        if len(sample_ids) >= MAX_SAMPLES:
            break

    if len(sample_ids) == 0:
        print('No samples collected. Check EXP paths or caches:', GNNE_LOC, OUR_LOC)
        return

    # sort for readability
    order = list(range(len(sample_ids)))
    if SORT_BY.lower() == 'gt':
        order = sorted(order, key=lambda i: s_gt[i])

    xs = np.arange(len(order))

    s_gt_s = np.array([s_gt[i] for i in order], dtype=np.float64)
    s_gn_s_map = {r: np.array([s_gn_map[r][i] for i in order], dtype=np.float64) for r in RATIOS}
    s_our_s = np.array([s_our[i] for i in order], dtype=np.float64)
    gea_gn_s_map = {r: np.array([gea_gn_map[r][i] for i in order], dtype=np.float64) for r in RATIOS}
    gea_our_s = np.array([gea_our[i] for i in order], dtype=np.float64)

    # metrics summary
    # choose a reference ratio (closest to 0.2 if present) for summary text
    ref_r = min(RATIOS, key=lambda x: abs(x - 0.2))
    mean_gap_gn = float(np.mean(np.abs(s_gn_s_map[ref_r] - s_gt_s)))
    mean_gap_our = float(np.mean(np.abs(s_our_s - s_gt_s)))
    mean_gea_gn = float(np.mean(gea_gn_s_map[ref_r]))
    mean_gea_our = float(np.mean(gea_our_s))

    # helpers for smoothing and binning
    def moving_mean_std(arr: np.ndarray, w: int):
        n = len(arr)
        if n == 0:
            return arr, np.zeros_like(arr)
        w = max(1, min(int(w), n))
        if w % 2 == 0:
            w = max(1, w - 1)
        if w <= 1:
            return arr, np.zeros_like(arr)
        pad = (w // 2, w // 2)
        a_pad = np.pad(arr, pad, mode='edge')
        k = np.ones(w, dtype=np.float64) / w
        mu = np.convolve(a_pad, k, mode='valid')
        a2_pad = np.pad(arr * arr, pad, mode='edge')
        mu2 = np.convolve(a2_pad, k, mode='valid')
        var = np.maximum(mu2 - mu * mu, 0.0)
        std = np.sqrt(var)
        return mu, std

    def compute_bins_by_chunks(sorted_arrays, B: int):
        n = len(next(iter(sorted_arrays.values())))
        B = max(1, min(int(B), n))
        edges = [int(np.ceil(i * n / B)) for i in range(B + 1)]
        centers = []
        means = {k: [] for k in sorted_arrays.keys()}
        stds = {k: [] for k in sorted_arrays.keys()}
        for b in range(B):
            s, e = edges[b], edges[b + 1]
            if e <= s:
                e = min(n, s + 1)
            centers.append((s + e - 1) / 2.0)
            for k, arr in sorted_arrays.items():
                seg = arr[s:e]
                means[k].append(float(np.mean(seg)))
                stds[k].append(float(np.std(seg)))
        x = np.arange(1, B + 1)
        return x, centers, means, stds

    # plotting (left: main curves; right: 2x2 explanation panels)
    plt.figure(figsize=(8, 6), dpi=200)
    # Increase bottom (GEA) subplot height by adjusting height ratios
    gs_main = plt.GridSpec(2, 2, width_ratios=[2, 3], height_ratios=[1, 0.8], wspace=0.15, hspace=0.12)

    ax1 = plt.subplot(gs_main[0, 0])
    ax2 = plt.subplot(gs_main[1, 0], sharex=ax1)

    # color palette for multiple ratios
    import matplotlib as mpl
    cmap = plt.cm.Blues
    nR = max(1, len(RATIOS))
    colors = {r: cmap(0.3 + 0.6 * (i / max(1, nR - 1))) for i, r in enumerate(RATIOS)}

    mode = PLOT_MODE.lower()
    if mode == 'line':
        ax1.plot(xs, s_gt_s, color='black', lw=1.5, label='Ground Truth')
        for r in RATIOS:
            ax1.plot(xs, s_gn_s_map[r], color=colors[r], lw=1.2, label=f'GNNE r={r:g}')
        ax1.plot(xs, s_our_s, color='#d08770', lw=1.2, label='OUR')
        for r in RATIOS:
            yg = np.clip(gea_gn_s_map[r], 0.0, 1.0)
            ax2.plot(xs, yg, color=colors[r], lw=1.2, label=f'GNNE r={r:g}')
        y_our = np.clip(gea_our_s, 0.0, 1.0)
        ax2.plot(xs, y_our, color='#d08770', lw=1.2, label='OUR')
    elif mode == 'ribbon':
        mu_gt, sd_gt = moving_mean_std(s_gt_s, SMOOTH_WIN)
        mu_our, sd_our = moving_mean_std(s_our_s, SMOOTH_WIN)
        ax1.plot(xs, mu_gt, color='black', lw=1.6, label='Ground Truth')
        ax1.fill_between(xs, mu_gt - sd_gt, mu_gt + sd_gt, color='black', alpha=SHADE_ALPHA, linewidth=0)
        for r in RATIOS:
            mu_gn, sd_gn = moving_mean_std(s_gn_s_map[r], SMOOTH_WIN)
            ax1.plot(xs, mu_gn, color=colors[r], lw=1.4, label=f'GNNE r={r:g}')
            ax1.fill_between(xs, mu_gn - sd_gn, mu_gn + sd_gn, color=colors[r], alpha=SHADE_ALPHA, linewidth=0)
        ax1.plot(xs, mu_our, color='#d08770', lw=1.4, label='OUR')
        ax1.fill_between(xs, mu_our - sd_our, mu_our + sd_our, color='#d08770', alpha=SHADE_ALPHA, linewidth=0)
        mu_gea_our, sd_gea_our = moving_mean_std(gea_our_s, SMOOTH_WIN)
        for r in RATIOS:
            mu_gea_gn, sd_gea_gn = moving_mean_std(gea_gn_s_map[r], SMOOTH_WIN)
            mu_gn_c = np.clip(mu_gea_gn, 0.0, 1.0)
            lo = np.clip(mu_gea_gn - sd_gea_gn, 0.0, 1.0)
            hi = np.clip(mu_gea_gn + sd_gea_gn, 0.0, 1.0)
            ax2.plot(xs, mu_gn_c, color=colors[r], lw=1.4, label=f'GNNE r={r:g}')
            ax2.fill_between(xs, lo, hi, color=colors[r], alpha=SHADE_ALPHA, linewidth=0)
        mu_our_c = np.clip(mu_gea_our, 0.0, 1.0)
        lo = np.clip(mu_gea_our - sd_gea_our, 0.0, 1.0)
        hi = np.clip(mu_gea_our + sd_gea_our, 0.0, 1.0)
        ax2.plot(xs, mu_our_c, color='#d08770', lw=1.4, label='OUR')
        ax2.fill_between(xs, lo, hi, color='#d08770', alpha=SHADE_ALPHA, linewidth=0)
    elif mode == 'bins' and SORT_BY.lower() == 'gt':
        arrays = {'gt': s_gt_s, 'our': s_our_s, 'gea_our': gea_our_s}
        for r in RATIOS:
            arrays[f'gn@{r:g}'] = s_gn_s_map[r]
            arrays[f'gea_gn@{r:g}'] = gea_gn_s_map[r]
        xbins, centers, means, stds = compute_bins_by_chunks(arrays, BIN_NUM)
        ax1.errorbar(xbins, means['gt'], yerr=stds['gt'], fmt='-o', color='black', ms=4, lw=1.2, label='Ground Truth (bin)')
        jitter = np.linspace(-0.15, 0.15, num=max(2, len(RATIOS))) if len(RATIOS) > 1 else [0.0]
        for j, r in enumerate(RATIOS):
            ax1.errorbar(xbins + jitter[j], means[f'gn@{r:g}'], yerr=stds[f'gn@{r:g}'], fmt='-o', color=colors[r], ms=4, lw=1.2, label=f'GNNE r={r:g}')
        ax1.errorbar(xbins, means['our'], yerr=stds['our'], fmt='-o', color='#d08770', ms=4, lw=1.2, label='OUR')

        for j, r in enumerate(RATIOS):
            m = np.clip(np.asarray(means[f'gea_gn@{r:g}'], dtype=float), 0.0, 1.0)
            s = np.asarray(stds[f'gea_gn@{r:g}'], dtype=float)
            yerr_low = np.minimum(s, m - 0.0)
            yerr_up = np.minimum(s, 1.0 - m)
            ax2.errorbar(xbins + jitter[j], m, yerr=[yerr_low, yerr_up], fmt='-o', color=colors[r], ms=4, lw=1.2, label=f'GNNE r={r:g}')
        m = np.clip(np.asarray(means['gea_our'], dtype=float), 0.0, 1.0)
        s = np.asarray(stds['gea_our'], dtype=float)
        yerr_low = np.minimum(s, m - 0.0)
        yerr_up = np.minimum(s, 1.0 - m)
        ax2.errorbar(xbins, m, yerr=[yerr_low, yerr_up], fmt='-o', color='#d08770', ms=4, lw=1.2, label='OUR (bin)')
        ax2.set_xticks(xbins)
        ax2.set_xlabel('GT sparsity bins (low → high)')
    elif mode == 'delta':
        d_our = s_our_s - s_gt_s
        mu_our, sd_our = moving_mean_std(d_our, SMOOTH_WIN)
        ax1.axhline(0.0, color='black', lw=1.0, ls=':')
        for r in RATIOS:
            d_gn = s_gn_s_map[r] - s_gt_s
            mu_gn, sd_gn = moving_mean_std(d_gn, SMOOTH_WIN)
            ax1.plot(xs, mu_gn, color=colors[r], lw=1.4, label=f'GNNE Δs r={r:g}')
            ax1.fill_between(xs, mu_gn - sd_gn, mu_gn + sd_gn, color=colors[r], alpha=SHADE_ALPHA, linewidth=0)
        ax1.plot(xs, mu_our, color='#d08770', lw=1.4, label='OUR Δs')
        ax1.fill_between(xs, mu_our - sd_our, mu_our + sd_our, color='#d08770', alpha=SHADE_ALPHA, linewidth=0)

        mu_gea_our, sd_gea_our = moving_mean_std(gea_our_s, SMOOTH_WIN)
        for r in RATIOS:
            mu_gea_gn, sd_gea_gn = moving_mean_std(gea_gn_s_map[r], SMOOTH_WIN)
            mu_gn_c = np.clip(mu_gea_gn, 0.0, 1.0)
            lo = np.clip(mu_gea_gn - sd_gea_gn, 0.0, 1.0)
            hi = np.clip(mu_gea_gn + sd_gea_gn, 0.0, 1.0)
            ax2.plot(xs, mu_gn_c, color=colors[r], lw=1.4, label=f'GNNE r={r:g}')
            ax2.fill_between(xs, lo, hi, color=colors[r], alpha=SHADE_ALPHA, linewidth=0)
        mu_our_c = np.clip(mu_gea_our, 0.0, 1.0)
        lo = np.clip(mu_gea_our - sd_gea_our, 0.0, 1.0)
        hi = np.clip(mu_gea_our + sd_gea_our, 0.0, 1.0)
        ax2.plot(xs, mu_our_c, color='#d08770', lw=1.4, label='OUR')
        ax2.fill_between(xs, lo, hi, color='#d08770', alpha=SHADE_ALPHA, linewidth=0)
    else:
        ax1.plot(xs, s_gt_s, color='black', lw=1.5, label='Ground Truth')
        for r in RATIOS:
            ax1.plot(xs, s_gn_s_map[r], color=colors[r], lw=1.2, label=f'GNNE r={r:g}')
        ax1.plot(xs, s_our_s, color='#d08770', lw=1.2, label='OUR')
        for r in RATIOS:
            yg = np.clip(gea_gn_s_map[r], 0.0, 1.0)
            ax2.plot(xs, yg, color=colors[r], lw=1.2, label=f'GNNE r={r:g}')
        y_our = np.clip(gea_our_s, 0.0, 1.0)
        ax2.plot(xs, y_our, color='#d08770', lw=1.2, label='OUR')

    # cosmetics
    # ax1.set_title(f'Per-sample sparsity & GEA — {DATASET} test set({len(xs)} samples)')
    ax1.set_ylabel('Sparsity' if mode != 'delta' else 'ΔSparsity (method - GT)')
    if not (mode == 'bins' and SORT_BY.lower() == 'gt'):
        ax1.set_xlim(-0.5, len(xs) - 0.5)
    ax2.set_xlabel('Sample (sorted by GT sparsity)' if SORT_BY.lower() == 'gt' else 'Sample index')
    if mode != 'delta':
        ax1.set_ylim(0, 0.8)
    else:
        ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, ls='--', alpha=0.3)
    # Build custom legend for sparsity: order = [GNNE..., GT, OUR], layout 3 rows x 2 cols (cap GNNE count)
    LEGEND_MAX_GNNE = int(os.environ.get('SAMPLE_LEGEND_MAX_GNNE', '4'))
    h_s, l_s = ax1.get_legend_handles_labels()
    items_s = list(zip(l_s, h_s))
    gn_s = [(lab, hd) for (lab, hd) in items_s if lab.startswith('GNNE')]
    gn_s.sort(key=lambda x: x[0])
    if LEGEND_MAX_GNNE > 0:
        gn_s = gn_s[:LEGEND_MAX_GNNE]
    gt_s = [(lab, hd) for (lab, hd) in items_s if lab.startswith('Ground Truth')]
    our_s = [(lab, hd) for (lab, hd) in items_s if lab.startswith('OUR')]
    labs_s = [x[0] for x in gn_s] + [x[0] for x in gt_s] + [x[0] for x in our_s]
    hnds_s = [x[1] for x in gn_s] + [x[1] for x in gt_s] + [x[1] for x in our_s]
    if labs_s:
        ax1.legend(hnds_s, labs_s, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, columnspacing=0.6, handlelength=2)

    ax2.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.6)
    ax2.set_ylabel('GEA')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, ls='--', alpha=0.3)
    # Build custom legend for GEA: order = [GNNE..., OUR], layout 3 rows x 2 cols
    h_g, l_g = ax2.get_legend_handles_labels()
    items_g = list(zip(l_g, h_g))
    gn_g = [(lab, hd) for (lab, hd) in items_g if lab.startswith('GNNE')]
    gn_g.sort(key=lambda x: x[0])
    if LEGEND_MAX_GNNE > 0:
        gn_g = gn_g[:LEGEND_MAX_GNNE]
    our_g = [(lab, hd) for (lab, hd) in items_g if lab.startswith('OUR')]
    labs_g = [x[0] for x in gn_g] + [x[0] for x in our_g]
    hnds_g = [x[1] for x in gn_g] + [x[1] for x in our_g]
    if labs_g:
        ax2.legend(hnds_g, labs_g, frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0.01, 0.25), fontsize=8, columnspacing=0.6, handlelength=2)

    # Keep only first and last ticks on both axes for a cleaner look
    try:
        if mode == 'bins' and SORT_BY.lower() == 'gt':
            # Use bin indices range
            xt0, xt1 = (xbins[0], xbins[-1]) if 'xbins' in locals() and len(xbins) > 0 else (0, 1)
        else:
            xt0, xt1 = (int(xs[0]), int(xs[-1])) if len(xs) > 0 else (0, 1)
        ax1.set_xticks([xt0, xt1])
        ax2.set_xticks([xt0, xt1])
    except Exception:
        pass

    # Y ticks: keep only ends according to current limits
    for ax in (ax1, ax2):
        y0, y1 = ax.get_ylim()
        ax.set_yticks([y0, y1])

    # ----- Right side 2x2 panels: pick low/high GT sparsity samples with OUR GEA=1 -----
    eligible = [i for i in range(len(sample_ids)) if gea_our[i] >= 0.999]
    idx_low = idx_high = None
    if len(eligible) > 0:
        # Sort eligible indices by GT sparsity
        eligible_sorted_asc = sorted(eligible, key=lambda i: s_gt[i])
        eligible_sorted_desc = list(reversed(eligible_sorted_asc))
        # Pick by rank (1-based); clamp within range
        li = max(0, min(LOW_RANK - 1, len(eligible_sorted_asc) - 1))
        hi = max(0, min(HIGH_RANK - 1, len(eligible_sorted_desc) - 1))
        i_low = eligible_sorted_asc[li]
        i_high = eligible_sorted_desc[hi]
        idx_low = sample_ids[i_low]
        idx_high = sample_ids[i_high]

    def undirected_edge_set(edge_index: torch.Tensor):
        ei = edge_index.cpu().numpy()
        Eset = set()
        for c in range(ei.shape[1]):
            u = int(ei[0, c]); v = int(ei[1, c])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            Eset.add((a, b))
        return Eset

    def compute_layout(G: nx.Graph, layout_name: str, seed=None):
        name = (layout_name or '').lower()
        # Prefer layouts that reduce crossings; fall back safely
        try:
            if name in ('kamada', 'kk', 'kamada_kawai'):
                # Use seeded spring as initialization so different seeds yield different (but stable) layouts
                pos0 = nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G)
                return nx.kamada_kawai_layout(G, pos=pos0)
            if name in ('circular', 'circle'):
                return nx.circular_layout(G)
            if name in ('spectral',):
                return nx.spectral_layout(G)
            if name in ('spring', 'fr'):
                return nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G)
        except Exception:
            pass
        # Fallback chain
        for fn in (
            lambda: nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=seed) if seed is not None else None),
            lambda: nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G),
            lambda: nx.circular_layout(G),
            lambda: nx.spectral_layout(G),
        ):
            try:
                return fn()
            except Exception:
                continue
        # Last resort: positions on a line
        return {n: (i, 0.0) for i, n in enumerate(G.nodes())}

    def mask_to_edge_set(mask_bool: torch.Tensor, edge_index: torch.Tensor):
        ei = edge_index.cpu().numpy()
        m = mask_bool.detach().to('cpu').numpy().astype(bool)
        Eset = set()
        idxs = np.where(m)[0]
        for c in idxs:
            u = int(ei[0, c]); v = int(ei[1, c])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            Eset.add((a, b))
        return Eset

    def draw_sample_pair(ax_gt_vs_our, ax_gt_vs_gn, data, gt_mask, mask_our, mask_gn, title_prefix: str):
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        E_all = undirected_edge_set(data.edge_index)
        G.add_edges_from(E_all)
        # layout (default: kamada-kawai; configurable via SAMPLE_LAYOUT); controlled by SAMPLE_LAYOUT_SEED
        pos = compute_layout(G, LAYOUT, seed=LAYOUT_SEED)
        GT = mask_to_edge_set(gt_mask, data.edge_index)
        OUR = mask_to_edge_set(mask_our, data.edge_index)
        GN = mask_to_edge_set(mask_gn, data.edge_index)
        base_col = "#4d4d4d"
        col_our = '#d08770'
        col_gn = '#5e81ac'

        # Prepare node visuals (colors + labels)
        atom_labels, atom_colors = infer_atom_labels_and_colors(data)

        def _draw(ax, other_set, other_col, title):
            ax.axis('off')
            node_color = atom_colors if COLORED_NODES else 'white'
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100, node_color=node_color, edgecolors='black', linewidths=0.6, alpha=NODE_ALPHA)
            # base edges (exclude only explainer edges; don't highlight GT to reduce clutter)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(E_all - other_set), width=0.8, edge_color=base_col, alpha=0.99)
            # Explainer edges: thick semi-transparent colored lines (overlay)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(other_set), width=3.2, edge_color=other_col, alpha=0.99)
            if SHOW_NODE_LABELS:
                labels_map = {i: atom_labels[i] for i in range(data.num_nodes)}
                texts = nx.draw_networkx_labels(G, pos, labels=labels_map, ax=ax, font_size=NODE_FONT_SIZE, font_color='black')
                # Optional: add white stroke to improve readability (disabled by default)
                if LABEL_STROKE and isinstance(texts, dict):
                    for t in texts.values():
                        try:
                            t.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])
                        except Exception:
                            pass
            ax.set_title(title, fontsize=10)

        _draw(ax_gt_vs_our, OUR, col_our, f"{title_prefix}: OUR")
        _draw(ax_gt_vs_gn, GN, col_gn, f"{title_prefix}: GNNE@{ref_r:g}")

    if idx_low is not None and idx_high is not None:
        right = gs_main[:, 1].subgridspec(2, 2, wspace=0.2, hspace=0.25)
        ax_r00 = plt.subplot(right[0, 0])  # low: GT vs OUR
        ax_r01 = plt.subplot(right[0, 1])  # low: GT vs GNNE
        ax_r10 = plt.subplot(right[1, 0])  # high: GT vs OUR
        ax_r11 = plt.subplot(right[1, 1])  # high: GT vs GNNE

        # low sample
        data_l, gt_exp_l = dataset[idx_low]
        data_l = data_l.to(device)
        gt_imp_l = merge_edge_imp(gt_exp_l, int(data_l.edge_index.size(1)), prefer_continuous=False)
        if gt_imp_l is not None:
            gt_mask_l = (gt_imp_l > 0).to(torch.bool).to(device)
            exp_our_l = exp_exists_graph(idx_low, path=OUR_LOC, get_exp=True)
            ei_our_l = merge_edge_imp(exp_our_l, int(data_l.edge_index.size(1)), prefer_continuous=False)
            exp_gn_l = exp_exists_graph(idx_low, path=GNNE_LOC, get_exp=True)
            ei_gn_l = merge_edge_imp(exp_gn_l, int(data_l.edge_index.size(1)), prefer_continuous=True)
            if ei_our_l is not None and ei_gn_l is not None:
                our_mask_l = (ei_our_l > 0).to(torch.bool).to(device)
                gn_mask_l = top_ratio_mask(ei_gn_l.to(device), ref_r)
                draw_sample_pair(ax_r00, ax_r01, data_l, gt_mask_l, our_mask_l, gn_mask_l, title_prefix="GT-sparse sample")

        # high sample
        data_h, gt_exp_h = dataset[idx_high]
        data_h = data_h.to(device)
        gt_imp_h = merge_edge_imp(gt_exp_h, int(data_h.edge_index.size(1)), prefer_continuous=False)
        if gt_imp_h is not None:
            gt_mask_h = (gt_imp_h > 0).to(torch.bool).to(device)
            exp_our_h = exp_exists_graph(idx_high, path=OUR_LOC, get_exp=True)
            ei_our_h = merge_edge_imp(exp_our_h, int(data_h.edge_index.size(1)), prefer_continuous=False)
            exp_gn_h = exp_exists_graph(idx_high, path=GNNE_LOC, get_exp=True)
            ei_gn_h = merge_edge_imp(exp_gn_h, int(data_h.edge_index.size(1)), prefer_continuous=True)
            if ei_our_h is not None and ei_gn_h is not None:
                our_mask_h = (ei_our_h > 0).to(torch.bool).to(device)
                gn_mask_h = top_ratio_mask(ei_gn_h.to(device), ref_r)
                draw_sample_pair(ax_r10, ax_r11, data_h, gt_mask_h, our_mask_h, gn_mask_h, title_prefix="GT-dense sample")

    # annotate summary
    # txt = f"ref r={ref_r:g} |s_gn-s_gt|: {mean_gap_gn:.3f}  |s_our-s_gt|: {mean_gap_our:.3f}\n" \
    #     f"mean GEA — GNNE@{ref_r:g}: {mean_gea_gn:.3f}, OUR: {mean_gea_our:.3f}"
    # ax1.text(0.01, 0.02, txt, transform=ax1.transAxes, ha='left', va='bottom', fontsize=9,
    #          bbox=dict(boxstyle='round', fc='white', ec='lightgray', alpha=0.8))

    rtag = '-'.join([str(r).replace('.', 'p') for r in RATIOS])
    out_path = os.path.join(OUT_DIR, f'fig1_sample_{GNNE_NAME}_rs{rtag}_vs_{OUR_NAME}_{PLOT_MODE}.pdf')
    plt.tight_layout()
    plt.savefig(out_path)
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
