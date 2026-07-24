# -*- coding: utf-8 -*-
"""
直接读取已保存的 SHAP 缓存，重新生成热图和排序图。

说明：
  1. 不重新加载模型权重，不重新计算 SHAP。
  2. 两个面板只需要 shap_merged_n*.npy 和 explain_idx_n*.npy。
  3. explain_idx 用于回到测试集标签，恢复 true_class_median 口径所需的 y_exp。
"""

from pathlib import Path

# 中文注释：缓存和输出文件统一使用当前项目路径配置。
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    SHAP_FIGURE7_PANEL_DIR,
    SHAP_FIGURE7A_PATH,
    SHAP_FIGURE7C_PATH,
    SHAP_FIGURE7AC_PATH,
)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import plot_shap_summary as fig7


# 中文注释：缓存目录使用完整路径；默认自动选择最新修改的一组缓存。
CACHE_DIR = Path(SHAP_FIGURE7_PANEL_DIR)
FIGURE7A_PATH = Path(SHAP_FIGURE7A_PATH)
FIGURE7C_PATH = Path(SHAP_FIGURE7C_PATH)
FIGURE7AC_PATH = Path(SHAP_FIGURE7AC_PATH)

# 中文注释：如需固定某一组缓存，把 None 改成样本数，例如 2163。
CACHE_N = None


def _cache_n_from_name(path):
    """中文注释：从 shap_merged_n2163.npy 这类文件名中解析样本数。"""
    stem = path.stem
    return int(stem.rsplit("_n", 1)[1])


def _find_cache_pair():
    """中文注释：找到一组 shap 缓存和对应的 explain_idx 缓存。"""
    shap_files = list(CACHE_DIR.glob("shap_merged_n*.npy"))
    idx_files = list(CACHE_DIR.glob("explain_idx_n*.npy"))
    idx_by_n = {_cache_n_from_name(path): path for path in idx_files}

    candidates = []
    for shap_path in shap_files:
        cache_n = _cache_n_from_name(shap_path)
        idx_path = idx_by_n.get(cache_n)
        if idx_path is not None:
            candidates.append((cache_n, shap_path, idx_path))

    if not candidates:
        raise FileNotFoundError(
            f"没有找到完整缓存：{CACHE_DIR}\\shap_merged_n*.npy 和 explain_idx_n*.npy"
        )

    if CACHE_N is not None:
        for cache_n, shap_path, idx_path in candidates:
            if cache_n == CACHE_N:
                return cache_n, shap_path, idx_path
        raise FileNotFoundError(f"没有找到 CACHE_N={CACHE_N} 对应的完整缓存。")

    # 中文注释：默认按 SHAP 文件修改时间选择最新一组。
    return max(candidates, key=lambda item: item[1].stat().st_mtime)


def _load_y_exp(explain_idx):
    """中文注释：用保存的测试集行号恢复每个解释样本的真实类别。"""
    _, _, _, y_test, unique_labels = fig7.load_data(
        fig7.TRAIN_FILE,
        fig7.TEST_FILE,
        fig7.TRAIN_MASK_FILE,
        fig7.TEST_MASK_FILE,
    )
    return np.asarray(y_test)[explain_idx], unique_labels


def _read_png_on_white(path):
    """中文注释：读取 PNG，并把透明通道合成到白底，方便后续拼接。"""
    image = mpimg.imread(path)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        image = image[:, :, :3] * alpha + (1.0 - alpha)
    return image[:, :, :3]


def combine_panel_a_c():
    """中文注释：把 SHAP 热图和排序图左右拼接，并让两张图垂直居中对齐。"""
    left_image = _read_png_on_white(FIGURE7A_PATH)
    right_image = _read_png_on_white(FIGURE7C_PATH)

    gap_px = 140
    output_height = max(left_image.shape[0], right_image.shape[0])
    output_width = left_image.shape[1] + gap_px + right_image.shape[1]
    canvas = np.ones((output_height, output_width, 3), dtype=np.float32)

    left_y = (output_height - left_image.shape[0]) // 2
    right_y = (output_height - right_image.shape[0]) // 2
    right_x = left_image.shape[1] + gap_px

    canvas[left_y:left_y + left_image.shape[0], :left_image.shape[1], :] = left_image
    canvas[right_y:right_y + right_image.shape[0],
           right_x:right_x + right_image.shape[1], :] = right_image

    plt.imsave(FIGURE7AC_PATH, canvas, dpi=1200)
    print(f"已保存左右拼接图：{FIGURE7AC_PATH}")


def main():
    cache_n, shap_path, idx_path = _find_cache_pair()
    print(f"读取 SHAP 缓存：{shap_path}")
    print(f"读取样本行号：{idx_path}")

    merged_shap = np.load(shap_path)
    explain_idx = np.load(idx_path)
    y_exp, unique_labels = _load_y_exp(explain_idx)

    if merged_shap.shape[1] != len(explain_idx):
        raise ValueError(
            f"缓存样本数不一致：merged_shap={merged_shap.shape[1]}, "
            f"explain_idx={len(explain_idx)}"
        )

    # 中文注释：保持原脚本排序图的 true-class median 统计口径。
    fig7.RANKING_IMPORTANCE_STAT = "true_class_median"

    print(f"开始绘制 SHAP 热图和排序图，缓存样本数 n={cache_n}。")
    fig7.plot_panel_a(merged_shap, unique_labels, str(CACHE_DIR), y_exp)
    fig7.plot_panel_c(merged_shap, str(CACHE_DIR), y_exp)
    combine_panel_a_c()
    print(f"完成，图片已保存到：{CACHE_DIR}")


if __name__ == "__main__":
    main()
