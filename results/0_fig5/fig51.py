import os
import re
import sys
import argparse
from typing import List, Tuple

import torch

from data.utils import get_dataset
from metrics.edge_exp_metrics import jac_edge_max


KNOWN_DATASETS = [
    "benzene",
    "alkanecarbonyl",
    "fluoridecarbonyl",
    "mutagenicity",
    "mutag",
    "house_triangle",
    "house_cycle",
    "triangle_grid",
]


def _infer_dataset_from_path(path: str) -> str:
    """Infer dataset name by searching known dataset keys in the path (case-insensitive)."""
    low = path.replace("\\", "/").lower()
    for name in KNOWN_DATASETS:
        if (
            f"/{name}/" in low
            or low.endswith(f"/{name}")
            or low.startswith(f"{name}/")
            or f"_{name}/" in low
        ):
            return name
    # Sometimes the dataset name sits right after model type in our results path layout
    # e.g. results/1_explanation_accuracy/GIN_3layer/triangle_grid/metrics_result/EXPS/subx
    segments = [seg for seg in low.split("/") if seg]
    for seg in segments:
        if seg in KNOWN_DATASETS:
            return seg
    raise ValueError(
        f"无法从路径推断数据集名称，请确认路径中包含以下之一: {', '.join(KNOWN_DATASETS)}"
    )


def _load_gt_for_index(dataset, idx: int):
    """Return (graph, gt_exp_list) for given index using dataset's __getitem__."""
    g, gt_exp = dataset[idx]
    return g, gt_exp


def _iter_exp_files(folder: str) -> List[str]:
    files = []
    for fn in os.listdir(folder):
        if fn.endswith(".pt") and fn.startswith("exp_"):
            files.append(os.path.join(folder, fn))
    return files


def _extract_index_from_filename(filepath: str) -> int:
    m = re.search(r"exp_(\d+)\.pt$", os.path.basename(filepath))
    if not m:
        raise ValueError(f"无法从文件名中解析索引: {filepath}")
    return int(m.group(1))


def print_rank_idx(
    floder: str, dataset_name_override: str | None = None, seed: int = 0
) -> List[int]:
    """
    读取传入文件夹内的解释缓存(exp_*.pt)，计算每个解释的 jac_edge_max，按高到低排序，输出排序后的解释文件名。

    返回值: 排序后的索引列表（由文件名 exp_XXXXX.pt 提取出的整数，如 4412、131、637）。
    """
    folder = floder
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"不存在的目录: {folder}")

    dataset_name = (
        dataset_name_override.lower()
        if dataset_name_override
        else _infer_dataset_from_path(folder)
    )
    # 统一首字母大小写以匹配 get_dataset 内部分支
    # get_dataset 会做 .lower()，因此这里直接传递原样字符串也可；为可读性保持原名风格
    canonical = {
        "benzene": "Benzene",
        "alkanecarbonyl": "AlkaneCarbonyl",
        "fluoridecarbonyl": "FluorideCarbonyl",
        "mutagenicity": "Mutagenicity",
        "mutag": "MUTAG",
        "house_triangle": "house_triangle",
        "house_cycle": "house_cycle",
        "triangle_grid": "triangle_grid",
    }[dataset_name]

    # 加载数据集以获取每个图的 GT 解释
    device = "cpu"  # 只做度量，无需模型与 GPU
    dataset = get_dataset(canonical, seed=seed, device=device)

    # 遍历目录下的 exp_*.pt
    files = _iter_exp_files(folder)
    if not files:
        print(f"目录中未找到解释缓存 (.pt): {folder}")
        return []

    scored: List[Tuple[str, float]] = []
    for f in files:
        try:
            idx = _extract_index_from_filename(f)
            _, gt_exp = _load_gt_for_index(dataset, idx)
            # 先加载解释对象；为避免类解析问题，确保 Explanation 可用
            exp = torch.load(open(f, "rb"), map_location="cpu")
            score = jac_edge_max(gt_exp, exp)
            if score is None:
                score = float("-inf")
            scored.append((f, float(score)))
        except Exception as e:
            # 遇到坏文件：将其置于最低分，并打印提示
            print(f"跳过文件(错误): {os.path.basename(f)} -> {e}")
            scored.append((f, float("-inf")))

    # 分数降序；同分按文件名升序
    scored.sort(key=lambda x: (-x[1], os.path.basename(x[0])))

    # 打印排序后的文件名（不含路径）
    for f, _ in scored:
        print(os.path.basename(f))

    # 返回排序后的索引列表
    return [_extract_index_from_filename(f) for f, _ in scored]


def main():
    parser = argparse.ArgumentParser(description="按 jac_edge_max 排序解释缓存文件")
    parser.add_argument("folder", type=str, help="包含 exp_*.pt 的目录路径")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称(可选，覆盖自动推断)，如 triangle_grid/house_cycle/house_triangle/Benzene/Mutagenicity/MUTAG/AlkaneCarbonyl/FluorideCarbonyl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="构建数据集缓存的随机种子(与训练/缓存一致更稳妥)",
    )
    args = parser.parse_args()
    print_rank_idx(args.folder, dataset_name_override=args.dataset, seed=args.seed)


if __name__ == "__main__":
    dataset_name = "FluorideCarbonyl"  # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle
    model_type = "GIN_3layer"  # GIN_3layer, GCN_3layer, GAT_2layer
    exp_name = "coaex"

    exp_loc = os.path.join(
        "results",
        "1_explanation_accuracy",
        f"{model_type}",
        f"{dataset_name}",
        "metrics_result",
        "EXPS",
        f"{exp_name}",
    )
    l = print_rank_idx(exp_loc, seed=42)

    output_path = os.path.join(
        os.path.dirname(__file__), f"sorted_indices{model_type}{dataset_name}.txt"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in l:
            f.write(f"{idx}\n")
