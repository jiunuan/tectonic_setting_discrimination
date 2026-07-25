from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import re
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_MAJOR_NORM_CSV,
    ARCHEAN_POOL_CSV,
    ZENODO_ARCHEAN_CSV,
    ARCHEAN_CONSISTENCY_DIR,
)

from archean_s3_preprocess import preprocess_archean

# 中文注释：当前项目通过集中配置选择正式数据或等价的 Zenodo 发布表。
OUTPUT_DIR = Path(ARCHEAN_CONSISTENCY_DIR)
TRAIN_CSV_PATH = Path(TRAIN_MAJOR_NORM_CSV)
ARCHEAN_CSV_PATH = Path(ZENODO_ARCHEAN_CSV if Path(ZENODO_ARCHEAN_CSV).exists() else ARCHEAN_POOL_CSV)
FIG_HARKER_PATH = OUTPUT_DIR / "fig_harker_train_vs_archean.png"
FIG_CLASSIC_PATH = OUTPUT_DIR / "fig_classic_discrimination_train_vs_archean.png"
FIG_RATIO_PATH = OUTPUT_DIR / "fig_ratio_density_train_vs_archean.png"
RATIO_SUMMARY_PATH = OUTPUT_DIR / "ratio_density_summary.csv"
REPORT_PATH = OUTPUT_DIR / "distribution_consistency_report.md"
APPENDIX_PATH = OUTPUT_DIR / "appendix_training_application_distribution_consistency.md"

# =========================
# 视觉风格
# =========================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "legend.fontsize": 13,
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.linestyle": (0, (4, 3)),
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "grid.color": "#D9D9D9",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "ps.fonttype": 42,
    }
)

COLOR_TRAIN = "#6F9FC9"
COLOR_ARCHEAN = "#C76E4A"
COLOR_TEXT = "#000000"
COLOR_GRID = "#D9D9D9"
COLOR_REFERENCE = "#777777"
COLOR_SPINE = "#000000"
TECTONIC_COLUMN = "TECTONICSETTING"
CFB_LABEL = "CONTINENTAL FLOOD BASALT"
CFB_TARGET_COUNT = 6920
ARCHEAN_TARGET_COUNT = 3012
RANDOM_SEED = 42


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：去单位、去特殊字符并转为大写。"""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        clean = re.sub(r"\(.*?\)", "", str(col))
        clean = re.sub(r"[^0-9A-Za-z]+", "", clean).upper()
        rename_map[col] = clean
    return df.rename(columns=rename_map)


def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """读取并清洗单个数据集。"""
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    df["DATASET"] = dataset_name
    return df


def load_current_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取现代训练集和太古代应用集，保持与原分布检查脚本一致的样品筛选。"""
    train = load_dataset(TRAIN_CSV_PATH, "Modern training")
    if TECTONIC_COLUMN not in train.columns:
        raise ValueError(f"Modern training data missing column: {TECTONIC_COLUMN}")

    # 中文注释：固定随机种子保留 6920 条 CFB，使训练集规模与正式流程一致。
    train[TECTONIC_COLUMN] = train[TECTONIC_COLUMN].astype(str).str.strip().str.upper()
    train = train.loc[train[TECTONIC_COLUMN].ne("NAN")].reset_index(drop=True)
    cfb_indices = np.flatnonzero(train[TECTONIC_COLUMN].to_numpy() == CFB_LABEL)
    if len(cfb_indices) < CFB_TARGET_COUNT:
        raise ValueError(f"Only {len(cfb_indices)} CFB samples, fewer than {CFB_TARGET_COUNT}.")
    selected_cfb = np.sort(
        np.random.default_rng(RANDOM_SEED).choice(
            cfb_indices,
            size=CFB_TARGET_COUNT,
            replace=False,
        )
    )
    non_cfb_indices = np.flatnonzero(train[TECTONIC_COLUMN].to_numpy() != CFB_LABEL)
    keep_indices = np.sort(np.concatenate([non_cfb_indices, selected_cfb]))
    train = train.iloc[keep_indices].reset_index(drop=True)

    # 中文注释：太古代应用集沿用 preprocess_archean 的正式筛选逻辑。
    archean_raw = pd.read_csv(ARCHEAN_CSV_PATH, low_memory=False)
    archean = preprocess_archean(
        archean_raw,
        expected_sample_count=ARCHEAN_TARGET_COUNT,
        dataset_name="Harker Archean application",
    )
    archean = normalize_columns(archean)
    archean["DATASET"] = "Archean application"
    return train, archean


def to_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """将目标列强制转成数值，方便绘图。"""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def finite_values(*arrays: np.ndarray) -> np.ndarray:
    """拼接多个数组中的有限值。"""
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def display_sample(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """对现代训练集散点做固定随机抽样，避免点云过密。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) <= max_points:
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), max_points, replace=False)
    return x[idx], y[idx]


def style_axes(ax: plt.Axes, *, log_grid: bool = False) -> None:
    """统一坐标轴外观。"""
    # 中文注释：classic 两联图保留四条边界线，颜色和 PCA 图一致，线宽固定为 0.6。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLOR_SPINE)
        spine.set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", length=3, width=0.7, color=COLOR_SPINE, labelcolor=COLOR_SPINE)
    ax.grid(True, which="major", axis="both", color=COLOR_GRID, alpha=0.65, linewidth=0.5, linestyle=(0, (4, 3)))
    if log_grid:
        ax.grid(True, which="minor", axis="both", color=COLOR_GRID, alpha=0.25, linewidth=0.35, linestyle=(0, (1, 3)))


def use_plain_log_tick_labels(ax: plt.Axes) -> None:
    """log 坐标使用普通数字刻度标签。"""
    formatter = mticker.FuncFormatter(lambda value, _pos: f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    inside_upper_right: bool = False,
) -> None:
    """添加子图编号。a/c/e 放图内右上角，b/d/f 放图内左上角，并保持同一水平高度。"""
    if inside_upper_right:
        ax.text(
            0.965,
            0.98,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=24,
            fontweight="bold",
            color=COLOR_TEXT,
            clip_on=False,
            zorder=30,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.6),
        )
    else:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=24,
            fontweight="bold",
            color=COLOR_TEXT,
            clip_on=False,
            zorder=30,
        )


def add_log_ellipse(
    ax: plt.Axes,
    *,
    center: tuple[float, float],
    width_log: float,
    height_log: float,
    angle: float,
    **kwargs,
) -> None:
    """在 log-log 坐标中用多边形近似椭圆参考域。"""
    theta = np.linspace(0, 2 * np.pi, 200)
    radians = np.deg2rad(angle)
    x0, y0 = np.log10(center[0]), np.log10(center[1])
    x = width_log / 2 * np.cos(theta)
    y = height_log / 2 * np.sin(theta)
    xr = x * np.cos(radians) - y * np.sin(radians)
    yr = x * np.sin(radians) + y * np.cos(radians)
    coords = np.column_stack([10 ** (x0 + xr), 10 ** (y0 + yr)])
    ax.add_patch(mpatches.Polygon(coords, closed=True, **kwargs))


def build_classic_figure(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """绘制 classic 判别图；当前只保留 a/b 两个面板，c 面板暂时停用。"""
    modern_color = COLOR_TRAIN
    archean_color = COLOR_ARCHEAN
    panel_labels = ["a", "b"]
    label_box = dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.60)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))

    # -------------------------
    # Ti-V 图
    # -------------------------
    ax = axes[0]
    ti_train = train["TIO2"].to_numpy(dtype=float) * 5994.0 / 1000.0
    ti_arc = archean["TIO2"].to_numpy(dtype=float) * 5994.0 / 1000.0
    v_train = train["V"].to_numpy(dtype=float)
    v_arc = archean["V"].to_numpy(dtype=float)
    ti_train_display, v_train_display = display_sample(ti_train, v_train, 10000, RANDOM_SEED + 10)

    ax.scatter(ti_train_display, v_train_display, s=5.0, c=modern_color, alpha=0.24, linewidths=0, rasterized=True, zorder=1, label="Modern training")
    ax.scatter(ti_arc, v_arc, s=20, c=archean_color, alpha=0.88, edgecolors="white", linewidths=0.28, rasterized=True, zorder=3, label="Archean application")

    x_max = max(np.nanpercentile(finite_values(ti_train, ti_arc), 99.5), 1.0)
    y_max = max(np.nanpercentile(finite_values(v_train, v_arc), 99.5), 1.0)
    x_line = np.linspace(max(0.0, np.nanpercentile(finite_values(ti_train, ti_arc), 0.5)), x_max, 300)
    y_20 = 1000.0 * x_line / 20.0
    y_50 = 1000.0 * x_line / 50.0
    y_100 = 1000.0 * x_line / 100.0
    y_top = y_max * 1.02
    # 中文注释：按 Ti/V = 20、50、100 正确裁剪阴影带，避免右侧高 Ti 区域被 where 条件误删。
    ax.fill_between(x_line, np.minimum(y_20, y_top), y_top, where=y_20 < y_top, color="#E7E2D6", alpha=0.28, zorder=0)
    ax.fill_between(x_line, y_50, np.minimum(y_20, y_top), where=y_50 < y_top, color="#DDE8EE", alpha=0.30, zorder=0)
    ax.fill_between(x_line, y_100, np.minimum(y_50, y_top), where=y_100 < y_top, color="#E9EEF1", alpha=0.24, zorder=0)
    for ratio in [20, 50, 100]:
        y_line = 1000.0 * x_line / ratio
        ax.plot(x_line, y_line, linestyle=(0, (4, 3)), color=COLOR_REFERENCE, linewidth=1.2, alpha=0.90, zorder=2)

    # 中文注释：Ti/V<20 的低 Ti/高 V 场对应 IAT/arc tholeiite；MORB/BABB 放在 Ti/V=20-50 中间场。
    ax.annotate("IAT", xy=(4.0, 360), xytext=(3.35, 545), fontsize=14, color="#333333", bbox=label_box, arrowprops=dict(arrowstyle="-", color=COLOR_REFERENCE, lw=0.55, alpha=0.65))
    ax.annotate("MORB / BABB", xy=(10.0, 300), xytext=(14.0, 450), fontsize=14, color="#333333", bbox=label_box, arrowprops=dict(arrowstyle="-", color=COLOR_REFERENCE, lw=0.55, alpha=0.65))
    ax.annotate("OIB", xy=(14.0, 220), xytext=(18.8, 280), fontsize=14, color="#333333", bbox=label_box, arrowprops=dict(arrowstyle="-", color=COLOR_REFERENCE, lw=0.55, alpha=0.65))
    ax.set_title("Ti-V", fontsize=16, fontweight="normal", pad=6)
    ax.set_xlabel("Ti/1000 (ppm)")
    ax.set_ylabel("V (ppm)")
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_ylim(0.0, y_max * 1.02)
    style_axes(ax)
    add_panel_label(ax, panel_labels[0])

    scatter_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=modern_color, markeredgecolor="none", markersize=7, alpha=0.55, label="Modern training"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=archean_color, markeredgecolor="white", markeredgewidth=0.5, markersize=8.6, alpha=0.90, label="Archean application"),
    ]
    legend = ax.legend(handles=scatter_handles, loc="upper right", frameon=True, framealpha=0.92, fontsize=14, borderpad=0.45, handletextpad=0.45, labelspacing=0.32)
    legend.get_frame().set_edgecolor("#D0D0D0")
    legend.get_frame().set_linewidth(0.5)

    # -------------------------
    # Th/Yb-Nb/Yb 图
    # -------------------------
    ax = axes[1]
    nb_train_raw = train["NB"].to_numpy(dtype=float)
    yb_train_raw = train["YB"].to_numpy(dtype=float)
    th_train_raw = train["TH"].to_numpy(dtype=float)
    nb_arc_raw = archean["NB"].to_numpy(dtype=float)
    yb_arc_raw = archean["YB"].to_numpy(dtype=float)
    th_arc_raw = archean["TH"].to_numpy(dtype=float)

    mask_train = np.isfinite(nb_train_raw) & np.isfinite(yb_train_raw) & np.isfinite(th_train_raw) & (nb_train_raw > 0) & (yb_train_raw > 0) & (th_train_raw > 0)
    mask_arc = np.isfinite(nb_arc_raw) & np.isfinite(yb_arc_raw) & np.isfinite(th_arc_raw) & (nb_arc_raw > 0) & (yb_arc_raw > 0) & (th_arc_raw > 0)
    nb_yb_train = nb_train_raw[mask_train] / yb_train_raw[mask_train]
    th_yb_train = th_train_raw[mask_train] / yb_train_raw[mask_train]
    nb_yb_arc = nb_arc_raw[mask_arc] / yb_arc_raw[mask_arc]
    th_yb_arc = th_arc_raw[mask_arc] / yb_arc_raw[mask_arc]
    nb_yb_train_display, th_yb_train_display = display_sample(nb_yb_train, th_yb_train, 10000, RANDOM_SEED + 11)

    ax.scatter(nb_yb_train_display, th_yb_train_display, s=5.0, c=modern_color, alpha=0.24, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(nb_yb_arc, th_yb_arc, s=20, c=archean_color, alpha=0.88, edgecolors="white", linewidths=0.28, rasterized=True, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100.0)
    ax.set_ylim(0.01, 10.0)
    use_plain_log_tick_labels(ax)

    array_x = np.logspace(np.log10(0.1), np.log10(100.0), 300)
    ax.fill_between(array_x, np.maximum(0.028 * array_x, 0.01), np.minimum(0.18 * array_x, 10.0), color="#6E6E6E", alpha=0.28, zorder=0)
    add_log_ellipse(ax, center=(0.55, 0.30), width_log=0.72, height_log=1.20, angle=-28, facecolor="#7A7A7A", edgecolor="#333333", alpha=0.34, zorder=0.2)
    add_log_ellipse(ax, center=(1.75, 1.15), width_log=0.72, height_log=1.22, angle=-28, facecolor="#E6E6E6", edgecolor="#333333", alpha=0.58, zorder=0.25)
    ax.text(0.32, 0.36, "OA", fontsize=14, ha="center", va="center", color="#222222", zorder=1)
    ax.text(1.38, 2.22, "CA", fontsize=14, ha="center", va="center", color="#222222", zorder=1)

    x_line = np.logspace(np.log10(0.1), np.log10(100.0), 300)
    for th_nb in [0.05, 0.10, 0.30]:
        ax.plot(x_line, th_nb * x_line, linestyle=(0, (4, 3)), color=COLOR_REFERENCE, linewidth=1.2, alpha=0.90, zorder=2)
    ax.annotate("MORB-OIB array", xy=(0.32, 0.055), xytext=(0.14, 0.022), fontsize=14, color="#333333", bbox=label_box, arrowprops=dict(arrowstyle="-", color=COLOR_REFERENCE, lw=0.55, alpha=0.65))

    ax.set_title("Th/Yb-Nb/Yb", fontsize=16, fontweight="normal", pad=6)
    ax.set_xlabel("Nb/Yb")
    ax.set_ylabel("Th/Yb", labelpad=-8)
    style_axes(ax, log_grid=True)
    add_panel_label(ax, panel_labels[1])

    # 中文注释：原 Zr/Y-Zr 的 c 面板暂时停用，不创建 axes[2]，避免 classic 图出现第三个面板。
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94], w_pad=0.35)
    FIG_CLASSIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_CLASSIC_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def robust_range(*arrays: np.ndarray, lower: float = 0.5, upper: float = 99.5, pad: float = 0.06) -> tuple[float, float]:
    """根据分位数估计稳定坐标范围。"""
    values = finite_values(*arrays)
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """生成一维高斯核，用于无 SciPy 环境下的轻量平滑。"""
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _smooth_2d(values: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """对二维直方图做可复现的 separable Gaussian 平滑。"""
    kernel = _gaussian_kernel1d(sigma)
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=0, arr=values)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), axis=1, arr=out)
    return out


def normalized_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    bins: int = 180,
    bw_adjust: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算归一化二维 Gaussian KDE 网格。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    grid_x = np.linspace(x_range[0], x_range[1], bins)
    grid_y = np.linspace(y_range[0], y_range[1], bins)
    if x.size < 5:
        return grid_x, grid_y, np.zeros((bins, bins), dtype=float)
    xx, yy = np.meshgrid(grid_x, grid_y)
    # 中文注释：使用真正的二维 Gaussian KDE，避免直方图平滑造成块状密度面。
    kde = gaussian_kde(np.vstack([x, y]))
    kde.set_bandwidth(kde.factor * bw_adjust)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    max_value = float(np.nanmax(density)) if density.size else 0.0
    if max_value > 0:
        density = density / max_value
    return grid_x, grid_y, density


def smooth_density(values: np.ndarray, *, grid_size: int = 1000, bw_adjust: float = 2.25) -> tuple[np.ndarray, np.ndarray]:
    """用一维 Gaussian KDE 生成平滑密度曲线。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, grid_size)
    if values.size < 3 or np.nanstd(values) == 0:
        return grid, np.zeros_like(grid)
    # 中文注释：用更密的采样网格和略宽带宽，避免 ratio 密度曲线出现折线感。
    kde = gaussian_kde(values)
    kde.set_bandwidth(kde.factor * bw_adjust)
    density = kde(grid)
    return grid, density


def ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """计算正值比值。"""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    ratio = pd.Series(np.nan, index=num.index, dtype=float)
    mask = num.gt(0) & den.gt(0) & num.notna() & den.notna()
    ratio.loc[mask] = num.loc[mask] / den.loc[mask]
    return ratio


def build_harker_figure(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """重新生成 Harker 主量元素分布图，使用手动 axes 布局保证每行左右子图严格水平对齐。"""
    panels = [
        ("MGO", "TIO2", "MgO (wt.%)", "TiO$_2$ (wt.%)", "TiO$_2$-MgO"),
        ("MGO", "FEOT", "MgO (wt.%)", "FeO$_T$ (wt.%)", "FeO$_T$-MgO"),
        ("MGO", "AL2O3", "MgO (wt.%)", "Al$_2$O$_3$ (wt.%)", "Al$_2$O$_3$-MgO"),
    ]
    panel_labels = "abcdef"

    # 中文注释：右侧 KDE 采用 imshow 连续光晕叠加。
    # 色图最高值直接使用左侧散点同一套 COLOR_TRAIN 和 COLOR_ARCHEAN。
    modern_density_cmap = LinearSegmentedColormap.from_list(
        "modern_density_soft_glow",
        ["#F7FAFD", "#DDEAF4", "#BBD4E9", COLOR_TRAIN],
    )
    archean_density_cmap = LinearSegmentedColormap.from_list(
        "archean_density_soft_glow",
        ["#FCF4F0", "#F1CFC0", "#DEA184", COLOR_ARCHEAN],
    )

    # 中文注释：四条轻量环线只提示太古代 KDE 形态，不强化硬边界。
    archean_contour_levels = [0.25, 0.45, 0.65, 0.85]

    # 中文注释：不要用 tight_layout / constrained_layout / subplots_adjust。
    # 这里直接用 fig.add_axes 固定每个坐标轴的位置。
    # 每一行左右两个 axes 使用完全相同的 bottom 和 height，
    # 因而 a-b、c-d、e-f 的图框上下边界会严格水平对齐。
    fig = plt.figure(figsize=(13.2, 13.6))
    axes = np.empty((3, 2), dtype=object)

    fig_left = 0.075
    fig_right = 0.985
    fig_bottom = 0.070
    fig_top = 0.955
    col_gap = 0.080
    row_gap = 0.090
    right_col_ratio = 0.92

    usable_width = fig_right - fig_left - col_gap
    left_width = usable_width / (1.0 + right_col_ratio)
    right_width = left_width * right_col_ratio
    x_positions = [fig_left, fig_left + left_width + col_gap]
    widths = [left_width, right_width]

    usable_height = fig_top - fig_bottom - 2.0 * row_gap
    ax_height = usable_height / 3.0

    for row in range(3):
        y0 = fig_top - (row + 1) * ax_height - row * row_gap
        for col in range(2):
            axes[row, col] = fig.add_axes(
                [x_positions[col], y0, widths[col], ax_height]
            )

    for row, (xcol, ycol, xlabel, ylabel, title) in enumerate(panels):
        scatter_ax = axes[row, 0]
        density_ax = axes[row, 1]

        x_train = train[xcol].to_numpy(dtype=float)
        y_train = train[ycol].to_numpy(dtype=float)
        x_arc = archean[xcol].to_numpy(dtype=float)
        y_arc = archean[ycol].to_numpy(dtype=float)

        valid_train = np.isfinite(x_train) & np.isfinite(y_train)
        valid_arc = np.isfinite(x_arc) & np.isfinite(y_arc)
        x_train = x_train[valid_train]
        y_train = y_train[valid_train]
        x_arc = x_arc[valid_arc]
        y_arc = y_arc[valid_arc]

        x_train_display, y_train_display = display_sample(
            x_train,
            y_train,
            6200,
            RANDOM_SEED + row,
        )

        x_lo, x_hi = robust_range(x_train, x_arc, lower=0.5, upper=99.5, pad=0.04)
        y_lo, y_hi = robust_range(y_train, y_arc, lower=0.5, upper=99.5, pad=0.08)
        x_lo = max(0.0, x_lo)
        y_lo = max(0.0, y_lo)

        scatter_ax.scatter(
            x_train_display,
            y_train_display,
            s=9.0,
            color=COLOR_TRAIN,
            alpha=0.28,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        scatter_ax.scatter(
            x_arc,
            y_arc,
            s=24.0,
            color=COLOR_ARCHEAN,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.24,
            rasterized=True,
            zorder=3,
        )

        grid_x_arc, grid_y_arc, density_arc = normalized_density_grid(
            x_arc,
            y_arc,
            (x_lo, x_hi),
            (y_lo, y_hi),
        )
        grid_x_train, grid_y_train, density_train = normalized_density_grid(
            x_train,
            y_train,
            (x_lo, x_hi),
            (y_lo, y_hi),
        )

        # 中文注释：现代训练集作为蓝色底层连续光晕。
        modern_alpha = np.clip(density_train, 0.0, 1.0) ** 0.60 * 0.92
        density_ax.imshow(
            density_train,
            origin="lower",
            extent=(x_lo, x_hi, y_lo, y_hi),
            cmap=modern_density_cmap,
            vmin=0.0,
            vmax=1.0,
            alpha=modern_alpha,
            aspect="auto",
            interpolation="bilinear",
            zorder=1,
        )

        # 中文注释：太古代应用集作为上层连续光晕。
        archean_alpha = np.clip(density_arc, 0.0, 1.0) ** 0.55 * 0.88
        density_ax.imshow(
            density_arc,
            origin="lower",
            extent=(x_lo, x_hi, y_lo, y_hi),
            cmap=archean_density_cmap,
            vmin=0.0,
            vmax=1.0,
            alpha=archean_alpha,
            aspect="auto",
            interpolation="bilinear",
            zorder=2,
        )

        # 中文注释：四圈 KDE 环线，颜色与左侧太古代散点一致。
        if np.nanmax(density_arc) >= min(archean_contour_levels):
            density_ax.contour(
                grid_x_arc,
                grid_y_arc,
                density_arc,
                levels=archean_contour_levels,
                colors=COLOR_ARCHEAN,
                linewidths=0.48,
                alpha=0.42,
                antialiased=True,
                zorder=3,
            )

        density_ax.scatter(
            np.nanmedian(x_train),
            np.nanmedian(y_train),
            marker="D",
            s=72,
            facecolors="white",
            edgecolors=COLOR_TRAIN,
            linewidths=1.45,
            zorder=5,
        )
        density_ax.scatter(
            np.nanmedian(x_arc),
            np.nanmedian(y_arc),
            marker="o",
            s=78,
            facecolors="white",
            edgecolors=COLOR_ARCHEAN,
            linewidths=1.45,
            zorder=6,
        )

        for col, ax in enumerate((scatter_ax, density_ax)):
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.set_title(title, fontsize=18, pad=8, fontweight="normal")
            ax.tick_params(axis="both", which="major", labelsize=16)
            style_axes(ax)

            label = panel_labels[row * 2 + col]
            add_panel_label(
                ax,
                label,
                inside_upper_right=(label in {"a", "c", "e"}),
            )

    scatter_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLOR_TRAIN,
            markeredgecolor="none",
            markersize=7.5,
            alpha=0.55,
            label="Modern training",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLOR_ARCHEAN,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8.0,
            alpha=0.90,
            label="Archean application",
        ),
    ]
    legend = axes[0, 0].legend(
        handles=scatter_handles,
        loc="upper right",
        bbox_to_anchor=(0.88, 0.99),
        frameon=True,
        framealpha=0.92,
        fontsize=15,
        borderpad=0.45,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    legend.get_frame().set_edgecolor("#D0D0D0")
    legend.get_frame().set_linewidth(0.6)

    center_handles = [
        mpatches.Patch(
            facecolor=COLOR_TRAIN,
            edgecolor="none",
            alpha=0.72,
            label="Modern density",
        ),
        mpatches.Patch(
            facecolor=COLOR_ARCHEAN,
            edgecolor=COLOR_ARCHEAN,
            linewidth=0.6,
            alpha=0.72,
            label="Archean density",
        ),
        (
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=COLOR_TRAIN,
                markeredgewidth=1.4,
                markersize=8.0,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=COLOR_ARCHEAN,
                markeredgewidth=1.4,
                markersize=8.0,
            ),
        ),
    ]
    center_legend = axes[0, 1].legend(
        handles=center_handles,
        labels=["Modern density", "Archean density", "Median centers"],
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=15,
        borderpad=0.45,
        handletextpad=0.45,
        labelspacing=0.35,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    center_legend.get_frame().set_edgecolor("#D0D0D0")
    center_legend.get_frame().set_linewidth(0.6)

    output_path = FIG_HARKER_PATH

    # 中文注释：这里不要再用 bbox_inches="tight"。
    # 你的 rcParams 里已经设置了 savefig.bbox="tight"，它会重新裁剪整张图，
    # 有时会让看起来固定的网格在导出后产生视觉错位。
    # bbox_inches=None 会覆盖全局 tight 设置，保留手动 axes 布局。
    fig.savefig(output_path, dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)


def build_ratio_figure(train: pd.DataFrame, archean: pd.DataFrame) -> pd.DataFrame:
    """重新生成判别比值密度图，并使用更大的标题、标签、刻度和图例字体。"""
    ratio_specs = [("Ba/Nb", "BA", "NB"), ("Th/Nb", "TH", "NB"), ("Nb/La", "NB", "LA"), ("Zr/Y", "ZR", "Y")]
    panel_labels = "abcd"
    summary_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.6))
    axes = axes.ravel()
    for idx, (ratio_name, num_col, den_col) in enumerate(ratio_specs):
        ax = axes[idx]
        train_ratio = ratio_series(train[num_col], train[den_col]).to_numpy(dtype=float)
        arc_ratio = ratio_series(archean[num_col], archean[den_col]).to_numpy(dtype=float)
        train_ratio = train_ratio[np.isfinite(train_ratio) & (train_ratio > 0)]
        arc_ratio = arc_ratio[np.isfinite(arc_ratio) & (arc_ratio > 0)]
        for dataset_name, values in (("Modern training", train_ratio), ("Archean application", arc_ratio)):
            summary_rows.append({"dataset": dataset_name, "ratio": ratio_name, "n_positive": int(values.size), "median": float(np.median(values)) if values.size else np.nan, "q25": float(np.percentile(values, 25)) if values.size else np.nan, "q75": float(np.percentile(values, 75)) if values.size else np.nan, "min": float(np.min(values)) if values.size else np.nan, "max": float(np.max(values)) if values.size else np.nan})
        x_train, y_train = smooth_density(np.log10(train_ratio))
        x_arc, y_arc = smooth_density(np.log10(arc_ratio))
        ax.plot(x_train, y_train, color=COLOR_TRAIN, linewidth=1.8, alpha=0.95, label="Modern training", zorder=2)
        ax.plot(x_arc, y_arc, color=COLOR_ARCHEAN, linewidth=2.3, alpha=1.0, label="Archean application", zorder=3)
        ax.set_xlabel(f"log10({ratio_name})", fontsize=18)
        ax.set_ylabel("Density" if idx in (0, 2) else "", fontsize=18, labelpad=8)
        ax.tick_params(axis="both", which="major", labelsize=16)
        style_axes(ax)
        add_panel_label(ax, panel_labels[idx])
        if x_train.size and x_arc.size:
            x_min = min(np.nanmin(x_train), np.nanmin(x_arc))
            x_max = max(np.nanmax(x_train), np.nanmax(x_arc))
            ax.set_xlim(x_min - 0.08 * (x_max - x_min), x_max + 0.08 * (x_max - x_min))
        if idx == 0:
            legend = ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=16, borderpad=0.45, handlelength=1.65, handletextpad=0.45, labelspacing=0.35)
            legend.get_frame().set_edgecolor("#D0D0D0")
            legend.get_frame().set_linewidth(0.6)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98], w_pad=1.4, h_pad=1.5)
    output_path = FIG_RATIO_PATH
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RATIO_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return summary
def build_report(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """写出简短说明，记录 c 面板当前停用。"""
    text = (
        "# Training-application distribution consistency\n\n"
        f"Modern training samples: {len(train)}\n\n"
        f"Archean application samples: {len(archean)}\n\n"
        "Classic discrimination figure currently contains panels a and b only; "
        "the original Zr/Y-Zr panel c is disabled for later restoration.\n"
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    APPENDIX_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    """主流程。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, archean = load_current_datasets()
    needed_columns = ["MGO", "TIO2", "FEOT", "AL2O3", "V", "NB", "YB", "TH", "BA", "LA", "ZR", "Y"]
    train = to_numeric_frame(train, needed_columns)
    archean = to_numeric_frame(archean, needed_columns)
    build_harker_figure(train, archean)
    build_classic_figure(train, archean)
    build_ratio_figure(train, archean)
    build_report(train, archean)
    print(f"Harker figure saved to: {FIG_HARKER_PATH}")
    print(f"Classic figure saved to: {FIG_CLASSIC_PATH}")
    print(f"Ratio figure saved to: {FIG_RATIO_PATH}")


if __name__ == "__main__":
    main()
