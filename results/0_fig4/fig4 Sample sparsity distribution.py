import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Workspace & Imports ----------------
# Resolve project root relative to this file (../.. from results/0_fig4)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from data.utils import get_dataset, filter_correct_data
from gnn.train_gnn_for_dataset import get_model
from graphxai.utils.performance.load_exp import exp_exists_graph
from graphxai.utils import Explanation
from graphxai.metrics.metrics_graph import graph_exp_acc_graph


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- Config via environment ----------------
# Dataset/model to load for indexing, GT, and shapes
DATASET = os.environ.get('FIG4_DATASET', 'Benzene')
MODEL = os.environ.get('FIG4_MODEL', 'GIN_3layer')

# Two explainer cache paths (relative to project root or absolute)
PATH1 = os.environ.get('FIG4_PATH1', os.path.join('results', '1_explanation_accuracy', MODEL, DATASET, 'metrics_result', 'EXPS', 'coaex'))
PATH2 = os.environ.get('FIG4_PATH2', os.path.join('results', '1_explanation_accuracy', MODEL, DATASET, 'metrics_result', 'EXPS', 'rcex_top_0.445'))

# Optional method labels; fallback to folder names
LABEL1 = os.environ.get('FIG4_LABEL1', '')
LABEL2 = os.environ.get('FIG4_LABEL2', '')

# Single ratio applied to continuous importance to binarize (for both methods if needed)
RATIO = float(os.environ.get('FIG4_RATIO', '0.44'))

# Sorting and sampling controls
SORT_BY = os.environ.get('FIG4_SORT', 'gt')  # 'gt' | 'idx'
MAX_SAMPLES = int(os.environ.get('FIG4_MAX', '2000'))
SKIP_EMPTY_GT = os.environ.get('FIG4_SKIP_EMPTY_GT', '1') not in ('0', 'false', 'False')
DEBUG = os.environ.get('FIG4_DEBUG', '0') not in ('0', 'false', 'False')

# Plot appearance
PLOT_MODE = os.environ.get('FIG4_PLOT_MODE', 'ribbon')  # 'ribbon' | 'line' | 'bins' | 'delta'
SMOOTH_WIN = int(os.environ.get('FIG4_SMOOTH_WIN', '11'))
BIN_NUM = int(os.environ.get('FIG4_BIN_NUM', '10'))
SHADE_ALPHA = float(os.environ.get('FIG4_SHADE_ALPHA', '0.25'))

# Output
OUT_DIR = os.path.join('results', '0_fig4')
os.makedirs(os.path.join(ROOT, OUT_DIR), exist_ok=True)


# ---------------- Helpers ----------------
def _to_abs(path_like: str) -> str:
	if os.path.isabs(path_like):
		return path_like
	return os.path.abspath(os.path.join(ROOT, path_like))


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


def merge_edge_imp(exp_obj, E: int, prefer_continuous: bool = False):
	"""Merge Explanation or list[Explanation] into a 1-D tensor of length E.
	- If prefer_continuous=True and a list is provided, use elementwise max as continuous importance.
	- Else, union boolean.
	Returns None on failure."""
	if exp_obj is None:
		return None
	# Explanation with edge_imp
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
			return torch.max(S, dim=0).values
		else:
			return (S > 0).any(dim=0).to(torch.float32)
	return None


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


def main():
	# ---- Resolve paths and labels ----
	p1 = _to_abs(PATH1)
	p2 = _to_abs(PATH2)
	name1 = LABEL1.strip() or os.path.basename(p1.rstrip('/\\'))
	name2 = LABEL2.strip() or os.path.basename(p2.rstrip('/\\'))

	# ---- Load dataset and model ----
	seed = 42
	dataset = get_dataset(DATASET, seed, device)
	model = get_model(dataset, MODEL, load_state=True)
	model.eval()

	# Indices of correctly predicted test samples
	_, _, test_idx = filter_correct_data(dataset, model)

	# Storage
	sample_ids = []
	s_gt, s_m1, s_m2 = [], [], []
	acc_m1, acc_m2 = [], []  # ACC/GEA values

	# Diagnostics counters
	total_candidates = 0
	pred_ok_count = 0
	both_exp_exist = 0
	non_empty_gt = 0
	after_all_filters = 0
	missing_exp_samples = []
	empty_gt_samples = []

	for idx in tqdm(test_idx, desc=f'Fig4 on {DATASET}: R={RATIO}'):
		data, gt_exp = dataset[idx]
		data = data.to(device)

		# ensure model predicts correctly (redundant safety) — use data.batch if present
		with torch.no_grad():
			batch_vec = getattr(data, 'batch', None)
			if batch_vec is None:
				# fallback: all nodes belong to graph 0
				batch_vec = torch.zeros((int(data.num_nodes),), dtype=torch.long, device=device)
			out, _ = model(data.x, data.edge_index, batch=batch_vec)
			pred = out.argmax(dim=1).item()
		total_candidates += 1
		if pred != data.y.item():
			continue
		pred_ok_count += 1

		M = int(data.edge_index.size(1))
		if M == 0:
			continue

		# Ground-truth edge importance (union if multiple components)
		gt_imp = merge_edge_imp(gt_exp, M, prefer_continuous=False)
		if gt_imp is None or gt_imp.numel() != M:
			continue
		gt_mask = (gt_imp > 0).to(torch.bool).to(device)
		k_gt = int(gt_mask.sum().item())
		if SKIP_EMPTY_GT and k_gt == 0:
			if DEBUG and len(empty_gt_samples) < 10:
				empty_gt_samples.append(int(idx))
			continue
		if k_gt > 0:
			non_empty_gt += 1
		spars_gt = k_gt / float(M)

		# Load two explanations
		exp1 = exp_exists_graph(int(idx), path=p1, get_exp=True)
		exp2 = exp_exists_graph(int(idx), path=p2, get_exp=True)
		if exp1 is None or exp2 is None:
			if DEBUG and len(missing_exp_samples) < 10:
				missing_exp_samples.append(int(idx))
			continue
		both_exp_exist += 1

		# Merge as continuous (prefer_continuous=True) then binarize by ratio if needed
		ei1 = merge_edge_imp(exp1, M, prefer_continuous=True)
		ei2 = merge_edge_imp(exp2, M, prefer_continuous=True)
		if ei1 is None or ei2 is None or ei1.numel() != M or ei2.numel() != M:
			continue
		ei1 = ei1.to(device)
		ei2 = ei2.to(device)

		# If tensors already look boolean, keep; else ratio-threshold
		def to_mask(ei: torch.Tensor) -> torch.Tensor:
			if ei.dtype == torch.bool:
				return ei
			# detect 0/1 floats as bool
			if torch.all((ei == 0) | (ei == 1)):
				return (ei > 0).to(torch.bool)
			return top_ratio_mask(ei, RATIO)

		mask1 = to_mask(ei1)
		mask2 = to_mask(ei2)
		k1 = int(mask1.sum().item())
		k2 = int(mask2.sum().item())
		spars1 = k1 / float(M)
		spars2 = k2 / float(M)

		# ACC/GEA against GT
		try:
			acc1 = graph_exp_acc_graph(gt_exp, Explanation(edge_imp=mask1, graph=data))
			acc2 = graph_exp_acc_graph(gt_exp, Explanation(edge_imp=mask2, graph=data))
			# Support both float or (feat,node,edge) legacy returns
			def _to_edge_acc(acc):
				if isinstance(acc, (list, tuple)):
					return acc[-1]
				return acc
			gea1 = _to_edge_acc(acc1)
			gea2 = _to_edge_acc(acc2)
		except Exception as e:
			if DEBUG:
				print('[Diag] ACC computation failed at idx', int(idx), 'err:', repr(e))
			continue

		sample_ids.append(int(idx))
		s_gt.append(spars_gt)
		s_m1.append(spars1)
		s_m2.append(spars2)
		acc_m1.append(float(gea1))
		acc_m2.append(float(gea2))

		after_all_filters += 1
		if len(sample_ids) >= MAX_SAMPLES:
			break

	if len(sample_ids) == 0:
		print('No samples collected. Check paths or caches:', p1, p2)
		print(f'[Diag] test_idx: {len(test_idx)} | candidates seen: {total_candidates} | pred_ok: {pred_ok_count} | both_exp_exist: {both_exp_exist} | non_empty_gt: {non_empty_gt}')
		if DEBUG:
			if missing_exp_samples:
				print('[Diag] First missing-exp idx examples (up to 10):', missing_exp_samples)
			if empty_gt_samples:
				print('[Diag] First empty-GT idx examples (up to 10):', empty_gt_samples)
		return

	# ---- Sort order ----
	order = list(range(len(sample_ids)))
	if SORT_BY.lower() == 'gt':
		order = sorted(order, key=lambda i: s_gt[i])

	xs = np.arange(len(order))
	s_gt_s = np.array([s_gt[i] for i in order], dtype=np.float64)
	s1_s = np.array([s_m1[i] for i in order], dtype=np.float64)
	s2_s = np.array([s_m2[i] for i in order], dtype=np.float64)
	a1_s = np.array([acc_m1[i] for i in order], dtype=np.float64)
	a2_s = np.array([acc_m2[i] for i in order], dtype=np.float64)

	# ---- Plot ----
	plt.figure(figsize=(5, 10), dpi=600)
	gs = plt.GridSpec(2, 1, height_ratios=[1, 0.9], hspace=0.12)
	ax_s = plt.subplot(gs[0, 0])
	ax_a = plt.subplot(gs[1, 0], sharex=ax_s)

	mode = PLOT_MODE.lower().strip()
	col1 = '#d08770'  # warm
	col2 = '#5e81ac'  # cool

	if mode == 'line':
		ax_s.plot(xs, s_gt_s, color='black', lw=1.6, label='Ground Truth')
		ax_s.plot(xs, s1_s, color=col1, lw=1.3, label=name1)
		ax_s.plot(xs, s2_s, color=col2, lw=1.3, label=name2)
		ax_a.plot(xs, np.clip(a1_s, 0.0, 1.0), color=col1, lw=1.3, label=name1)
		ax_a.plot(xs, np.clip(a2_s, 0.0, 1.0), color=col2, lw=1.3, label=name2)
	elif mode == 'ribbon':
		mu_gt, sd_gt = moving_mean_std(s_gt_s, SMOOTH_WIN)
		mu1, sd1 = moving_mean_std(s1_s, SMOOTH_WIN)
		mu2, sd2 = moving_mean_std(s2_s, SMOOTH_WIN)
		ax_s.plot(xs, mu_gt, color='black', lw=1.6, label='Ground Truth')
		ax_s.fill_between(xs, mu_gt - sd_gt, mu_gt + sd_gt, color='black', alpha=SHADE_ALPHA, linewidth=0)
		ax_s.plot(xs, mu1, color=col1, lw=1.4, label=name1)
		ax_s.fill_between(xs, mu1 - sd1, mu1 + sd1, color=col1, alpha=SHADE_ALPHA, linewidth=0)
		ax_s.plot(xs, mu2, color=col2, lw=1.4, label=name2)
		ax_s.fill_between(xs, mu2 - sd2, mu2 + sd2, color=col2, alpha=SHADE_ALPHA, linewidth=0)

		ma1, sa1 = moving_mean_std(a1_s, SMOOTH_WIN)
		ma2, sa2 = moving_mean_std(a2_s, SMOOTH_WIN)
		ax_a.plot(xs, np.clip(ma1, 0.0, 1.0), color=col1, lw=1.4, label=name1)
		ax_a.fill_between(xs, np.clip(ma1 - sa1, 0.0, 1.0), np.clip(ma1 + sa1, 0.0, 1.0), color=col1, alpha=SHADE_ALPHA, linewidth=0)
		ax_a.plot(xs, np.clip(ma2, 0.0, 1.0), color=col2, lw=1.4, label=name2)
		ax_a.fill_between(xs, np.clip(ma2 - sa2, 0.0, 1.0), np.clip(ma2 + sa2, 0.0, 1.0), color=col2, alpha=SHADE_ALPHA, linewidth=0)
	elif mode == 'bins' and SORT_BY.lower() == 'gt':
		arrays_s = {'gt': s_gt_s, name1: s1_s, name2: s2_s}
		arrays_a = {name1: a1_s, name2: a2_s}
		xbins, centers, means_s, stds_s = compute_bins_by_chunks(arrays_s, BIN_NUM)
		_, _, means_a, stds_a = compute_bins_by_chunks(arrays_a, BIN_NUM)

		ax_s.errorbar(xbins, means_s['gt'], yerr=stds_s['gt'], fmt='-o', color='black', ms=4, lw=1.2, label='Ground Truth (bin)')
		jitter = 0.12
		ax_s.errorbar(xbins - jitter, means_s[name1], yerr=stds_s[name1], fmt='-o', color=col1, ms=4, lw=1.2, label=name1)
		ax_s.errorbar(xbins + jitter, means_s[name2], yerr=stds_s[name2], fmt='-o', color=col2, ms=4, lw=1.2, label=name2)

		m1 = np.clip(np.asarray(means_a[name1], dtype=float), 0.0, 1.0)
		s1 = np.asarray(stds_a[name1], dtype=float)
		err1_low = np.minimum(s1, m1 - 0.0)
		err1_up = np.minimum(s1, 1.0 - m1)
		ax_a.errorbar(xbins - jitter, m1, yerr=[err1_low, err1_up], fmt='-o', color=col1, ms=4, lw=1.2, label=name1)

		m2 = np.clip(np.asarray(means_a[name2], dtype=float), 0.0, 1.0)
		s2 = np.asarray(stds_a[name2], dtype=float)
		err2_low = np.minimum(s2, m2 - 0.0)
		err2_up = np.minimum(s2, 1.0 - m2)
		ax_a.errorbar(xbins + jitter, m2, yerr=[err2_low, err2_up], fmt='-o', color=col2, ms=4, lw=1.2, label=name2)
		ax_a.set_xticks(xbins)
		ax_a.set_xlabel('GT sparsity bins (low → high)')
	elif mode == 'delta':
		d1 = s1_s - s_gt_s
		d2 = s2_s - s_gt_s
		mu1, sd1 = moving_mean_std(d1, SMOOTH_WIN)
		mu2, sd2 = moving_mean_std(d2, SMOOTH_WIN)
		ax_s.axhline(0.0, color='black', lw=1.0, ls=':')
		ax_s.plot(xs, mu1, color=col1, lw=1.4, label=f'{name1} Δs')
		ax_s.fill_between(xs, mu1 - sd1, mu1 + sd1, color=col1, alpha=SHADE_ALPHA, linewidth=0)
		ax_s.plot(xs, mu2, color=col2, lw=1.4, label=f'{name2} Δs')
		ax_s.fill_between(xs, mu2 - sd2, mu2 + sd2, color=col2, alpha=SHADE_ALPHA, linewidth=0)

		ma1, sa1 = moving_mean_std(a1_s, SMOOTH_WIN)
		ma2, sa2 = moving_mean_std(a2_s, SMOOTH_WIN)
		ax_a.plot(xs, np.clip(ma1, 0.0, 1.0), color=col1, lw=1.4, label=name1)
		ax_a.fill_between(xs, np.clip(ma1 - sa1, 0.0, 1.0), np.clip(ma1 + sa1, 0.0, 1.0), color=col1, alpha=SHADE_ALPHA, linewidth=0)
		ax_a.plot(xs, np.clip(ma2, 0.0, 1.0), color=col2, lw=1.4, label=name2)
		ax_a.fill_between(xs, np.clip(ma2 - sa2, 0.0, 1.0), np.clip(ma2 + sa2, 0.0, 1.0), color=col2, alpha=SHADE_ALPHA, linewidth=0)
	else:
		ax_s.plot(xs, s_gt_s, color='black', lw=1.6, label='Ground Truth')
		ax_s.plot(xs, s1_s, color=col1, lw=1.3, label=name1)
		ax_s.plot(xs, s2_s, color=col2, lw=1.3, label=name2)
		ax_a.plot(xs, np.clip(a1_s, 0.0, 1.0), color=col1, lw=1.3, label=name1)
		ax_a.plot(xs, np.clip(a2_s, 0.0, 1.0), color=col2, lw=1.3, label=name2)

	# Cosmetics
	if not (mode == 'bins' and SORT_BY.lower() == 'gt'):
		ax_s.set_xlim(-0.5, len(xs) - 0.5)
	ax_s.set_ylabel('Sparsity' if mode != 'delta' else 'ΔSparsity (method - GT)')
	ax_a.set_xlabel('Sample (sorted by GT sparsity)' if SORT_BY.lower() == 'gt' else 'Sample index')
	ax_a.set_ylabel('ACC')
	if mode != 'delta':
		ax_s.set_ylim(0, 0.8)
	else:
		ax_s.set_ylim(-1.0, 1.0)
	ax_a.set_ylim(0, 1.0)
	ax_s.grid(True, ls='--', alpha=0.3)
	ax_a.grid(True, ls='--', alpha=0.3)

	# Legends
	h_s, l_s = ax_s.get_legend_handles_labels()
	# keep order: GT, name1, name2
	order_labels = []
	for key in ('Ground Truth', name1, name2):
		for lab, h in zip(l_s, h_s):
			if lab == key:
				order_labels.append((lab, h))
				break
	if order_labels:
		ax_s.legend([h for _, h in order_labels], [l for l, _ in order_labels], frameon=False, ncol=3, loc='upper left', fontsize=8)

	h_a, l_a = ax_a.get_legend_handles_labels()
	od_a = []
	for key in (name1, name2):
		for lab, h in zip(l_a, h_a):
			if lab == key:
				od_a.append((lab, h))
				break
	if od_a:
		ax_a.legend([h for _, h in od_a], [l for l, _ in od_a], frameon=False, ncol=2, loc='lower left', fontsize=8)

	# Simplify ticks (first and last only)
	try:
		xt0, xt1 = (int(xs[0]), int(xs[-1])) if len(xs) > 0 else (0, 1)
		ax_s.set_xticks([xt0, xt1])
		ax_a.set_xticks([xt0, xt1])
	except Exception:
		pass
	for ax in (ax_s, ax_a):
		y0, y1 = ax.get_ylim()
		ax.set_yticks([y0, y1])

	# Save
	rtag = str(RATIO).replace('.', 'p')
	out_name = f"fig4_sparsity_acc_{name1}_vs_{name2}_r{rtag}_{PLOT_MODE}.pdf"
	out_path = os.path.join(ROOT, OUT_DIR, out_name)
	plt.tight_layout()
	plt.savefig(out_path)
	print('Saved:', out_path)


if __name__ == '__main__':
	main()

