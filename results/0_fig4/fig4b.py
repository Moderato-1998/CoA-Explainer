import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Workspace root (two levels up from this file)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.utils import get_dataset, filter_correct_data
from gnn.train_gnn_for_dataset import get_model
from graphxai.utils.performance.load_exp import exp_exists_graph


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- Config ----------------
DATASET = os.environ.get('FIG4B_DATASET', 'Benzene')
MODEL = os.environ.get('FIG4B_MODEL', 'GIN_3layer')

PATH1 = os.environ.get('FIG4B_PATH1', os.path.join('results', '1_explanation_accuracy', MODEL, DATASET, 'metrics_result', 'EXPS', 'coaex'))
PATH2 = os.environ.get('FIG4B_PATH2', os.path.join('results', '1_explanation_accuracy', MODEL, DATASET, 'metrics_result', 'EXPS', 'gnnex'))

LABEL1 = os.environ.get('FIG4B_LABEL1', '')
LABEL2 = os.environ.get('FIG4B_LABEL2', '')

# If explanation is continuous, binarize by this ratio; otherwise keep bool mask
RATIO = float(os.environ.get('FIG4B_RATIO', '0.2'))

# Filter and plotting options
ONLY_CORRECT = os.environ.get('FIG4B_ONLY_CORRECT', '1') not in ('0', 'false', 'False')
SKIP_EMPTY_GT = os.environ.get('FIG4B_SKIP_EMPTY_GT', '1') not in ('0', 'false', 'False')
MAX_SAMPLES = int(os.environ.get('FIG4B_MAX', '1000000'))
NBINS = int(os.environ.get('FIG4B_BINS', '40'))
KDE = os.environ.get('FIG4B_KDE', '0') not in ('0', 'false', 'False')
SHOW_GT = os.environ.get('FIG4B_SHOW_GT', '1') not in ('0', 'false', 'False')
CURVE_SIGMA = float(os.environ.get('FIG4B_CURVE_SIGMA', '1.2'))  # smoothing in bins units for curves
LOGX = os.environ.get('FIG4B_LOGX', '0') not in ('0', 'false', 'False')
DEBUG = os.environ.get('FIG4B_DEBUG', '0') not in ('0', 'false', 'False')

OUT_DIR = os.path.join(ROOT, 'results', '0_fig4')
os.makedirs(OUT_DIR, exist_ok=True)


def _to_abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(ROOT, p))


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
    m = torch.zeros_like(e, dtype=torch.bool)
    m[top_idx] = True
    return m


def merge_edge_imp(exp_obj):
    if exp_obj is None:
        return None
    if hasattr(exp_obj, 'edge_imp'):
        return exp_obj.edge_imp
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
        # union as boolean
        return (S > 0).any(dim=0).to(torch.float32)
    return None


def to_mask(ei: torch.Tensor, ratio: float) -> torch.Tensor:
    if ei is None:
        return None
    if ei.dtype == torch.bool:
        return ei
    if torch.all((ei == 0) | (ei == 1)):
        return (ei > 0).to(torch.bool)
    return top_ratio_mask(ei, ratio)


def collect_edge_counts(dataset, model, idx_iter, cache_path: str):
    counts = []
    miss, ok = 0, 0
    for idx in tqdm(idx_iter, desc=f'Collect from {os.path.basename(cache_path)}'):
        data, gt = dataset[idx]
        if ONLY_CORRECT:
            with torch.no_grad():
                batch_vec = getattr(data, 'batch', None)
                if batch_vec is None:
                    batch_vec = torch.zeros((int(data.num_nodes),), dtype=torch.long, device=device)
                out, _ = model(data.x.to(device), data.edge_index.to(device), batch=batch_vec.to(device))
                if out.argmax(dim=1).item() != data.y.item():
                    continue
        if SKIP_EMPTY_GT:
            M = int(data.edge_index.size(1))
            gt_ei = merge_edge_imp(gt)
            if gt_ei is None or gt_ei.numel() != M:
                continue
            if int((gt_ei > 0).sum().item()) == 0:
                continue

        exp = exp_exists_graph(int(idx), path=cache_path, get_exp=True)
        if exp is None:
            miss += 1
            continue
        ei = merge_edge_imp(exp)
        if ei is None:
            miss += 1
            continue
        m = to_mask(ei, RATIO)
        if m is None:
            miss += 1
            continue
        k = int(m.sum().item())
        counts.append(k)
        ok += 1
        if len(counts) >= MAX_SAMPLES:
            break
    if DEBUG:
        print(f'[Diag] {os.path.basename(cache_path)}: ok={ok}, miss={miss}, collected={len(counts)}')
    return counts


def collect_gt_counts(dataset, model, idx_iter):
    counts = []
    ok, miss = 0, 0
    for idx in tqdm(idx_iter, desc='Collect GT'):
        data, gt = dataset[idx]
        if ONLY_CORRECT:
            with torch.no_grad():
                batch_vec = getattr(data, 'batch', None)
                if batch_vec is None:
                    batch_vec = torch.zeros((int(data.num_nodes),), dtype=torch.long, device=device)
                out, _ = model(data.x.to(device), data.edge_index.to(device), batch=batch_vec.to(device))
                if out.argmax(dim=1).item() != data.y.item():
                    continue
        ei = merge_edge_imp(gt)
        if ei is None:
            miss += 1
            continue
        m = (ei > 0).to(torch.bool) if ei.dtype != torch.bool else ei
        k = int(m.sum().item())
        if SKIP_EMPTY_GT and k == 0:
            continue
        counts.append(k)
        ok += 1
        if len(counts) >= MAX_SAMPLES:
            break
    if DEBUG:
        print(f'[Diag] GT: ok={ok}, miss={miss}, collected={len(counts)}')
    return counts


def main():
    seed = 42
    dataset = get_dataset(DATASET, seed, device)
    model = get_model(dataset, MODEL, load_state=True)
    model.eval()

    _, _, test_idx = filter_correct_data(dataset, model)
    # We iterate over test_idx to match the typical evaluation split
    p1 = _to_abs(PATH1)
    p2 = _to_abs(PATH2)
    name1 = LABEL1.strip() or os.path.basename(p1.rstrip('/\\'))
    name2 = LABEL2.strip() or os.path.basename(p2.rstrip('/\\'))

    c1 = collect_edge_counts(dataset, model, test_idx, p1)
    c2 = collect_edge_counts(dataset, model, test_idx, p2)
    cgt = collect_gt_counts(dataset, model, test_idx) if SHOW_GT else []

    if len(c1) == 0 and len(c2) == 0 and len(cgt) == 0:
        print('No counts collected. Check caches:', p1, p2)
        return

    plt.figure(figsize=(7, 4.5), dpi=200)
    # Choose common bins
    all_counts = np.array((c1 or [0]) + (c2 or [0]) + (cgt or [0]), dtype=np.int64)
    xmin, xmax = int(all_counts.min()), int(all_counts.max())
    if LOGX:
        # use logarithmic bins for wide distributions
        xmin = max(1, xmin)
        bins = np.logspace(np.log10(xmin), np.log10(max(xmin+1, xmax)), num=max(5, NBINS))
    else:
        bins = np.linspace(xmin - 0.5, xmax + 0.5, num=max(5, NBINS))

    alpha = 0.5
    color1 = '#d08770'
    color2 = '#5e81ac'

    # Helper: histogram density curve via Gaussian smoothing in bin space
    def density_curve(samples, bins, logx=False, sigma_bins=1.0):
        if samples is None or len(samples) == 0:
            return None, None
        hist, edges = np.histogram(samples, bins=bins, density=True)
        if logx:
            centers = np.sqrt(edges[:-1] * edges[1:])  # geometric center for log bins
        else:
            centers = 0.5 * (edges[:-1] + edges[1:])
        if sigma_bins and sigma_bins > 0:
            radius = int(max(1, round(3 * sigma_bins)))
            xs = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (xs / sigma_bins) ** 2)
            kernel = kernel / np.sum(kernel)
            hist = np.convolve(hist, kernel, mode='same')
        return centers, hist

    # GT as histogram bars
    if len(cgt) > 0:
        plt.hist(cgt, bins=bins, alpha=0.55, color='black', label='Ground Truth', density=True, edgecolor='none')

    # Methods as smooth density curves
    if len(c1) > 0:
        x1, y1 = density_curve(c1, bins, logx=LOGX, sigma_bins=CURVE_SIGMA)
        if x1 is not None:
            plt.plot(x1, y1, color=color1, lw=2.0, label=name1)
    if len(c2) > 0:
        x2, y2 = density_curve(c2, bins, logx=LOGX, sigma_bins=CURVE_SIGMA)
        if x2 is not None:
            plt.plot(x2, y2, color=color2, lw=2.0, label=name2)

    # Optional KDE (simple Gaussian KDE via numpy/scipy would be heavier; skip by default)
    # Optional scipy KDE for methods (additional overlay)
    if KDE:
        try:
            from scipy.stats import gaussian_kde
            # Build evaluation grid in the same domain
            xx = np.logspace(np.log10(max(1, xmin)), np.log10(max(2, xmax)), 400) if LOGX else np.linspace(max(0, xmin), xmax, 400)
            if len(c1) > 1:
                kde1 = gaussian_kde(c1)
                plt.plot(xx, kde1(xx), color=color1, lw=1.0, alpha=0.8)
            if len(c2) > 1:
                kde2 = gaussian_kde(c2)
                plt.plot(xx, kde2(xx), color=color2, lw=1.0, alpha=0.8)
        except Exception as e:
            if DEBUG:
                print('[Diag] KDE failed:', repr(e))

    plt.xlabel('Number of edges in explanation subgraph')
    plt.ylabel('Density')
    plt.grid(True, ls='--', alpha=0.3)
    plt.legend(frameon=False)
    if LOGX:
        plt.xscale('log')

    tag_gt = '_withGT' if SHOW_GT else ''
    out_name = f"fig4b_edgecount_dist_{name1}_vs_{name2}{tag_gt}.pdf"
    out_path = os.path.join(OUT_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path)
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
