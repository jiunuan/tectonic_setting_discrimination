# -*- coding: utf-8 -*-
"""绘制六案例构造环境组成图和高弧信号年龄山脊图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ARCHEAN_CASE_DIR
from archean_s3_preprocess import CASE_STUDIES_ORDER, CASE_STUDY_TITLES


PREDICTIONS_DIR = Path(str(ARCHEAN_CASE_DIR)) / "predictions"
BARS_OUTPUT_PATH = PREDICTIONS_DIR / "fig_case_studies_bars.png"
RIDGELINE_OUTPUT_PATH = PREDICTIONS_DIR / "fig_case_studies_ridgeline.png"
COMBINED_OUTPUT_PATH = PREDICTIONS_DIR / "fig_case_studies_bars_ridgeline.png"

CASE_PREDICTION_PATHS = {
    "Isua": PREDICTIONS_DIR / "Isua_predictions.csv",
    "Pilbara": PREDICTIONS_DIR / "Pilbara_predictions.csv",
    "Ivisaartoq": PREDICTIONS_DIR / "Ivisaartoq_predictions.csv",
    "Norseman_Kambalda": PREDICTIONS_DIR / "Norseman_Kambalda_predictions.csv",
    "Abitibi": PREDICTIONS_DIR / "Abitibi_predictions.csv",
    "North_China_Craton": PREDICTIONS_DIR / "North_China_Craton_predictions.csv",
}


# 中文注释：边框灰色与太古代时间演化主图和 PCA 图保持一致。
SPINE_GRAY = "#000000"
# 中文注释：左侧弧比例小字与 PCA 图 c 的样品数标签颜色一致。
C_NLABEL = "#5F666D"

SCI_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "DejaVu Sans",
    ],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.6,
    "axes.edgecolor": SPINE_GRAY,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "ps.fonttype": 42,
}

CLASS_COLORS = {
    # 中文注释：低饱和暖色与KDE的杏橙填色、陶土红曲线保持协调。
    "Continental arc": "#C94A34",
    "Intra-oceanic arc": "#D9755B",
    "Island arc": "#E8A66D",
    "BACK-ARC_BASIN": "#7896AE",
    "SPREADING_CENTER": "#A3BFCA",
    "OCEANIC PLATEAU": "#729675",
    "OCEAN ISLAND": "#A7BD83",
    "CONTINENTAL FLOOD BASALT": "#916B82",
    "CONTINENTAL_RIFT": "#BDA3B3",
}
HEATMAP_ZERO_COLOR = "#FBFBF8"
HEATMAP_LOW_COLOR = "#EDF4F4"
HEATMAP_HIGH_COLOR = "#5E8F9A"

CLASS_ABBREVS = {
    "Continental arc": "CA",
    "Intra-oceanic arc": "IOA",
    "Island arc": "IA",
    "BACK-ARC_BASIN": "BAB",
    "SPREADING_CENTER": "MOR",
    "OCEANIC PLATEAU": "OP",
    "OCEAN ISLAND": "OI",
    "CONTINENTAL FLOOD BASALT": "CF",
    "CONTINENTAL_RIFT": "CR",
}
ARC_LIKE_THRESHOLD = 0.50
LEGEND_ORDER = ["CF", "CR", "OP", "IA", "BAB", "OI", "IOA", "CA", "MOR"]

RIDGE_PERIODS = [
    {"name": "Eoarchean", "start": 4.00, "end": 3.60, "color": "#C8D9EC"},
    {"name": "Paleoarchean", "start": 3.60, "end": 3.20, "color": "#F0CADA"},
    {"name": "Mesoarchean", "start": 3.20, "end": 2.80, "color": "#FAE5C2"},
    {"name": "Neoarchean", "start": 2.80, "end": 2.50, "color": "#CDE8C7"},
]
RIDGE_X_MIN_GA = 2.45
RIDGE_X_MAX_GA = 4.00
RIDGE_KDE_BANDWIDTH_GA = 0.06
RIDGE_LANE_FILL = 0.62
RIDGE_RUG_MAX_N = 45
RIDGE_FILL_COLOR = "#A9DCD6"
RIDGE_LINE_COLOR = "#2D8078"


def _configure_chinese_font() -> None:
    """按字体名称选择可用中文字体，不依赖特定操作系统路径。"""
    for family in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            current = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = [family] + current
            break
        except ValueError:
            continue


def _soften_color(color: str, white_fraction: float = 0.18) -> tuple[float, float, float]:
    """将类别颜色适度向白色混合。"""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - white_fraction) + white_fraction)



def _case_order_by_age(case_results: dict[str, pd.DataFrame]) -> list[str]:
    """按案例代表年龄从老到新排序。"""
    ordered = sorted(CASE_STUDIES_ORDER, key=lambda item: item[2], reverse=True)
    return [label for label, _, _ in ordered if label in case_results]


def _load_case_results() -> dict[str, pd.DataFrame]:
    """读取六个案例预测结果。"""
    results: dict[str, pd.DataFrame] = {}
    for case_label, _, _ in CASE_STUDIES_ORDER:
        input_path = CASE_PREDICTION_PATHS[case_label]
        if not input_path.exists():
            print(f"[警告] 缺少案例预测文件: {input_path}")
            continue
        results[case_label] = pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    return results


def _draw_compact_legend(fig: plt.Figure) -> None:
    """绘制仅包含颜色和简称的紧凑图例。"""
    abbrev_to_class = {abbr: name for name, abbr in CLASS_ABBREVS.items()}
    handles = [
        mpatches.Patch(
            facecolor=_soften_color(CLASS_COLORS[abbrev_to_class[abbr]]),
            edgecolor="none",
            label=abbr,
        )
        for abbr in LEGEND_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=9,
        frameon=False,
        fontsize=11,
        handlelength=1.15,
        handleheight=0.8,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def _apply_boxed_panel(ax: plt.Axes, linewidth: float = 0.6) -> None:
    """给子图加四边线框，形成更规整的期刊面板风格。"""
    # 中文注释：保留刻度在左侧和下侧，但让上下左右四条边框都可见。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_GRAY)
        spine.set_linewidth(linewidth)
        spine.set_capstyle("butt")
    ax.tick_params(top=False, right=False)


def _short_case_label(case_label: str) -> str:
    """为紧凑面板生成更易阅读的案例名称。"""
    if case_label == "Norseman_Kambalda":
        return "Norseman-\nKambalda"
    if case_label == "North_China_Craton":
        return "North China\nCraton"
    return CASE_STUDY_TITLES.get(case_label, case_label)


def _case_composition_matrix(
    case_results: dict[str, pd.DataFrame],
    cases: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """按固定类别顺序整理案例组成矩阵，便于绘制紧凑热力图。"""
    class_by_abbrev = {abbrev: name for name, abbrev in CLASS_ABBREVS.items()}
    class_names = [class_by_abbrev[abbrev] for abbrev in LEGEND_ORDER]
    counts_matrix = np.zeros((len(cases), len(class_names)), dtype=float)

    for row_index, case_label in enumerate(cases):
        counts = case_results[case_label]["pred_class_name"].value_counts()
        for col_index, class_name in enumerate(class_names):
            counts_matrix[row_index, col_index] = float(counts.get(class_name, 0))

    row_totals = counts_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_matrix = np.divide(
            counts_matrix * 100.0,
            row_totals,
            out=np.zeros_like(counts_matrix),
            where=row_totals > 0,
        )
    return counts_matrix, percent_matrix, class_names, LEGEND_ORDER


def _class_tinted_heatmap_rgba(
    percent_matrix: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """用单一灰青连续色带表示百分比大小，避免用色相编码类别。"""
    del class_names
    rgba = np.ones((*percent_matrix.shape, 4), dtype=float)
    zero_color = np.asarray(mcolors.to_rgb(HEATMAP_ZERO_COLOR), dtype=float)
    low_color = np.asarray(mcolors.to_rgb(HEATMAP_LOW_COLOR), dtype=float)
    high_color = np.asarray(mcolors.to_rgb(HEATMAP_HIGH_COLOR), dtype=float)
    finite_values = percent_matrix[np.isfinite(percent_matrix)]
    scale_max = max(float(finite_values.max()) if finite_values.size else 1.0, 1.0)

    for row_index in range(percent_matrix.shape[0]):
        for col_index in range(percent_matrix.shape[1]):
            percent = percent_matrix[row_index, col_index]
            if not np.isfinite(percent) or percent <= 0:
                rgba[row_index, col_index, :3] = zero_color
                continue
            intensity = np.sqrt(min(percent / scale_max, 1.0))
            rgba[row_index, col_index, :3] = (
                low_color * (1.0 - intensity)
                + high_color * intensity
            )
    return rgba
def plot_six_case_horizontal_bars(
    case_results: dict[str, pd.DataFrame],
    output_path: Path = BARS_OUTPUT_PATH,
) -> None:
    """绘制六联横向柱状图，比较各案例构造环境组成。"""
    cases = _case_order_by_age(case_results)
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 9.4))

    for ax, case_label in zip(axes.flat, cases):
        data = case_results[case_label]
        counts = data["pred_class_name"].value_counts()
        classes = [name for name in counts.index if int(counts[name]) > 0]
        values = np.asarray([int(counts[name]) for name in classes])
        labels = [CLASS_ABBREVS.get(name, name) for name in classes]
        colors = [_soften_color(CLASS_COLORS.get(name, "#888888")) for name in classes]

        y_positions = np.arange(len(classes))
        bars = ax.barh(
            y_positions,
            values,
            height=0.62,
            color=colors,
            edgecolor="white",
            linewidth=0.45,
        )
        max_value = max(int(values.max()), 1)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + max_value * 0.018,
                bar.get_y() + bar.get_height() / 2.0,
                str(int(value)),
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#303030",
            )

        arc_count = int(_high_arc_mask(data).sum())
        arc_percent = 100.0 * arc_count / len(data)
        age_ga = dict(
            (label, age) for label, _, age in CASE_STUDIES_ORDER
        )[case_label]
        ax.set_title(
            f"{CASE_STUDY_TITLES.get(case_label, case_label)}"
            f"  (~{age_ga:g} Ga; n={len(data)}; arc={arc_percent:.0f}%)",
            loc="left",
            fontsize=13.0,
            pad=5,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=12, fontweight="normal")
        ax.invert_yaxis()
        ax.set_xlim(0, max_value * 1.02)
        ax.set_xlabel("Sample count")
        ax.grid(False)
        _apply_boxed_panel(ax)

    for ax in axes.flat[len(cases):]:
        ax.set_axis_off()

    _draw_compact_legend(fig)
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.975,
        bottom=0.075,
        hspace=0.36,
        wspace=0.27,
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[六联柱状图] {output_path}")


def _case_age_ga(data: pd.DataFrame) -> pd.Series:
    """提取样品年龄，优先C_AGE，回退AGE，单位转换为Ga。"""
    age = pd.Series(np.nan, index=data.index, dtype=float)
    if "C_AGE" in data.columns:
        age = pd.to_numeric(data["C_AGE"], errors="coerce")
    if "AGE" in data.columns:
        age = age.fillna(pd.to_numeric(data["AGE"], errors="coerce"))
    return age / 1000.0


def _high_arc_mask(data: pd.DataFrame) -> pd.Series:
    """识别与 Fig. 5 GeoDAN arc fraction 一致的 arc-like 样品。"""
    arc_probability = pd.to_numeric(
        data.get("Arc_probability3", np.nan),
        errors="coerce",
    )
    return arc_probability.ge(ARC_LIKE_THRESHOLD)


def _gaussian_density(ages: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """使用固定带宽计算年龄核密度形状。"""
    density = np.zeros_like(grid)
    for age in ages:
        density += np.exp(
            -((grid - age) ** 2) / (2.0 * RIDGE_KDE_BANDWIDTH_GA ** 2)
        )
    return density


def plot_high_arc_ridgeline(
    case_results: dict[str, pd.DataFrame],
    output_path: Path = RIDGELINE_OUTPUT_PATH,
) -> None:
    """按区域代表年龄从老到新绘制高弧样品KDE山脊图。"""
    cases = _case_order_by_age(case_results)
    age_by_case = {label: age for label, _, age in CASE_STUDIES_ORDER}
    grid = np.linspace(RIDGE_X_MIN_GA, RIDGE_X_MAX_GA, 600)

    fig = plt.figure(figsize=(11.0, 4.2))
    ax = fig.add_axes([0.14, 0.13, 0.82, 0.68])

    for period in RIDGE_PERIODS:
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        ax.axvspan(
            x_low,
            x_high,
            color=period["color"],
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )

    for row_index, case_label in enumerate(cases):
        baseline = (len(cases) - 1) - row_index
        data = case_results[case_label]
        arc_mask = _high_arc_mask(data)
        ages = _case_age_ga(data)[arc_mask].dropna().to_numpy(dtype=float)
        ages = ages[np.isfinite(ages)]

        ax.hlines(
            baseline,
            RIDGE_X_MIN_GA,
            RIDGE_X_MAX_GA,
            color="#D0D0D0",
            linewidth=0.45,
            zorder=2,
        )
        density = (
            _gaussian_density(ages, grid)
            if ages.size
            else _gaussian_density(
                np.asarray([age_by_case[case_label]]),
                grid,
            )
        )
        if density.max() > 0:
            curve = baseline + RIDGE_LANE_FILL * density / density.max()
            ax.fill_between(
                grid,
                baseline,
                curve,
                color="#F7CFA5",
                alpha=0.72,
                linewidth=0,
                zorder=3,
            )
            ax.plot(grid, curve, color="#C94A34", linewidth=1.15, zorder=4)

        if 0 < ages.size <= RIDGE_RUG_MAX_N:
            ax.vlines(
                ages,
                baseline - 0.16,
                baseline - 0.045,
                color="#C94A34",
                linewidth=0.6,
                alpha=0.7,
                zorder=3,
            )

    ax.set_xlim(RIDGE_X_MAX_GA, RIDGE_X_MIN_GA)
    ax.set_ylim(-0.55, len(cases) - 1 + RIDGE_LANE_FILL + 0.55)
    ax.set_xticks(np.arange(2.5, 4.01, 0.1))
    ax.set_xlabel("Age (Ga)")
    ax.set_yticks([(len(cases) - 1) - index for index in range(len(cases))])
    ax.set_yticklabels(
        [
            f"{CASE_STUDY_TITLES.get(case, case)}  "
            f"(~{age_by_case[case]:g} Ga)"
            for case in cases
        ],
        fontsize=10,
        fontweight="bold",
    )
    ax.tick_params(axis="y", length=0, pad=6)
    _apply_boxed_panel(ax)
    ax.grid(False)

    transform = ax.get_xaxis_transform()
    for period in RIDGE_PERIODS:
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        ax.add_patch(
            mpatches.Rectangle(
                (x_low, 1.01),
                x_high - x_low,
                0.05,
                transform=transform,
                facecolor=period["color"],
                edgecolor=(1.0, 1.0, 1.0, 0.45),
                linewidth=0.45,
                clip_on=False,
                zorder=5,
            )
        )
        ax.text(
            (x_low + x_high) / 2.0,
            1.075,
            f"{period['name']}\n{x_high:.1f}-{max(x_low, 2.5):.1f} Ga",
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            linespacing=1.0,
        )

    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[高弧KDE山脊图] {output_path}")


def plot_case_studies_bars_ridgeline(
    case_results: dict[str, pd.DataFrame],
    output_path: Path = COMBINED_OUTPUT_PATH,
) -> None:
    """绘制左侧组成热力图、右侧高弧KDE山脊图的双栏主图。"""
    cases = _case_order_by_age(case_results)
    age_by_case = {label: age for label, _, age in CASE_STUDIES_ORDER}

    fig = plt.figure(figsize=(13.2, 6.65))
    outer_grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.72, 0.86],
        left=0.070,
        right=0.985,
        top=0.895,
        bottom=0.155,
        wspace=0.08,
    )

    # 中文注释：左栏改为单一热力图，减少六联柱状图造成的视觉跳转和小字负担。
    heat_ax = fig.add_subplot(outer_grid[0, 0])
    _counts_matrix, percent_matrix, class_names, class_labels = (
        _case_composition_matrix(case_results, cases)
    )
    heat_rgba = _class_tinted_heatmap_rgba(percent_matrix, class_names)
    heat_ax.imshow(
        heat_rgba,
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )


    heat_ax.set_xticks(np.arange(len(class_labels)))
    heat_ax.set_xticklabels(class_labels, fontsize=13.4, fontweight="normal")
    heat_ax.xaxis.tick_bottom()
    heat_ax.tick_params(axis="x", length=0, pad=6, labeltop=False, labelbottom=True)

    # 中文注释：地区名与弧比例分开绘制，便于把弧比例做成更轻的小号灰字。
    row_labels = []
    arc_labels = []
    for case_label in cases:
        data = case_results[case_label]
        arc_count = int(_high_arc_mask(data).sum())
        arc_percent = 100.0 * arc_count / len(data)
        row_labels.append(_short_case_label(case_label))
        arc_labels.append(f"arc={arc_percent:.0f}%")
    heat_ax.set_yticks(np.arange(len(cases)))
    heat_ax.set_yticklabels([])
    heat_ax.tick_params(axis="y", length=0, pad=8)
    for row_index, (row_label, arc_label) in enumerate(zip(row_labels, arc_labels)):
        # 中文注释：最后两个地区名更长，弧比例略下移，避免与地区名贴得太近。
        arc_label_y = row_index + (0.18 if row_index >= len(cases) - 2 else 0.12)
        heat_ax.text(
            -0.030,
            row_index - 0.12,
            row_label,
            transform=heat_ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=13.4,
            fontweight="normal",
            color="#000000",
            linespacing=1.2,
            clip_on=False,
            zorder=4,
        )
        heat_ax.text(
            -0.030,
            arc_label_y,
            arc_label,
            transform=heat_ax.get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=11.0,
            fontweight="normal",
            color=C_NLABEL,
            clip_on=False,
            zorder=4,
        )

    heat_ax.set_xticks(np.arange(-0.5, len(class_labels), 1.0), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, len(cases), 1.0), minor=True)
    heat_ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.72)
    heat_ax.tick_params(which="minor", bottom=False, left=False)
    heat_ax.set_xlim(-0.5, len(class_labels) - 0.5)
    heat_ax.set_ylim(len(cases) - 0.5, -0.5)
    heat_ax.set_xlabel("Tectonic settings", fontsize=13.6, fontweight="normal", labelpad=8)
    heat_ax.set_ylabel("")

    # 中文注释：仅标注有辨识度的比例，低占比交给色块保留模式信息。
    for row_index in range(percent_matrix.shape[0]):
        for col_index in range(percent_matrix.shape[1]):
            percent = percent_matrix[row_index, col_index]
            if percent < 5.0:
                continue
            face_color = heat_rgba[row_index, col_index, :3]
            luminance = (
                0.2126 * face_color[0]
                + 0.7152 * face_color[1]
                + 0.0722 * face_color[2]
            )
            heat_ax.text(
                col_index,
                row_index,
                f"{percent:.0f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if luminance < 0.53 else "#333333",
                zorder=3,
            )
    # 中文注释：保留四周边框，统一为0.6线宽。
    _apply_boxed_panel(heat_ax)

    # 中文注释：右栏集中展示各区域高弧样品的年龄分布。
    ridge_ax = fig.add_subplot(outer_grid[0, 1])
    # 中文注释：右栏与热力图主体对齐，避免组合图出现上下错位。
    ridge_position = ridge_ax.get_position()
    ridge_ax.set_position(
        [
            ridge_position.x0,
            ridge_position.y0,
            ridge_position.width,
            heat_ax.get_position().y1 - ridge_position.y0,
        ]
    )
    grid = np.linspace(RIDGE_X_MIN_GA, RIDGE_X_MAX_GA, 600)
    for period in RIDGE_PERIODS:
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        ridge_ax.axvspan(
            x_low,
            x_high,
            color=period["color"],
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )

    for row_index, case_label in enumerate(cases):
        baseline = (len(cases) - 1) - row_index
        data = case_results[case_label]
        ages = _case_age_ga(data)[_high_arc_mask(data)].dropna().to_numpy(dtype=float)
        ages = ages[np.isfinite(ages)]

        ridge_ax.hlines(
            baseline,
            RIDGE_X_MIN_GA,
            RIDGE_X_MAX_GA,
            color="#D0D0D0",
            linewidth=0.45,
            zorder=2,
        )
        density = (
            _gaussian_density(ages, grid)
            if ages.size
            else _gaussian_density(np.asarray([age_by_case[case_label]]), grid)
        )
        if density.max() > 0:
            curve = baseline + RIDGE_LANE_FILL * density / density.max()
            ridge_ax.fill_between(
                grid,
                baseline,
                curve,
                color=RIDGE_FILL_COLOR,
                alpha=0.38,
                linewidth=0,
                zorder=3,
            )
            ridge_ax.plot(
                grid,
                curve,
                color=RIDGE_LINE_COLOR,
                linewidth=1.25,
                zorder=4,
            )

    ridge_ax.set_xlim(RIDGE_X_MAX_GA, RIDGE_X_MIN_GA)
    ridge_y_min = -0.22
    period_bar_bottom = len(cases) - 1 + RIDGE_LANE_FILL + 0.15
    period_bar_height = 0.30
    ridge_y_max = period_bar_bottom + period_bar_height
    ridge_ax.set_ylim(ridge_y_min, ridge_y_max)
    ridge_ax.set_xticks(np.arange(2.6, 4.01, 0.2))
    ridge_ax.set_xlabel("Age (Ga)", fontsize=13.0)
    ridge_label_positions = []
    for index, case in enumerate(cases):
        baseline = (len(cases) - 1) - index
        # 中文注释：标签位于相邻两条横线之间；Isua位于顶部年代条与首条横线之间。
        if case == "Isua":
            label_y = (baseline + period_bar_bottom) / 2.0
        else:
            label_y = baseline + 0.5
        ridge_label_positions.append(label_y)
    ridge_ax.set_yticks(ridge_label_positions)
    # 中文注释：案例名称只保留在左侧热力图，右侧共享行序但不重复标签。
    ridge_ax.set_yticklabels([])
    ridge_ax.tick_params(axis="x", labelsize=13.4)
    ridge_ax.tick_params(axis="y", length=0, pad=0)
    # 中文注释：保留四周边框，统一为0.6线宽。
    _apply_boxed_panel(ridge_ax)
    ridge_ax.spines["left"].set_bounds(ridge_y_min, ridge_y_max)
    ridge_ax.spines["right"].set_bounds(ridge_y_min, ridge_y_max)
    ridge_ax.spines["top"].set_bounds(RIDGE_X_MIN_GA, RIDGE_X_MAX_GA)
    ridge_ax.spines["bottom"].set_bounds(RIDGE_X_MIN_GA, RIDGE_X_MAX_GA)
    ridge_ax.grid(False)

    # 中文注释：年代条放入坐标轴内部，利用第一条KDE上方空间。
    for period_index, period in enumerate(RIDGE_PERIODS):
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        # 中文注释：最左侧年代块略微内缩，避免与x轴竖线重合。
        rectangle_x_high = x_high - 0.003 if period_index == 0 else x_high
        ridge_ax.add_patch(
            mpatches.Rectangle(
                (x_low, period_bar_bottom),
                rectangle_x_high - x_low,
                period_bar_height,
                facecolor=period["color"],
                alpha=0.80,
                edgecolor=(1.0, 1.0, 1.0, 0.45),
                linewidth=0.45,
                clip_on=True,
                zorder=5,
            )
        )
        ridge_ax.text(
            (x_low + x_high) / 2.0,
            period_bar_bottom + period_bar_height / 2.0,
            period["name"],
            ha="center",
            va="center",
            fontsize=11.2,
            fontweight="semibold",
            color="#303030",
            clip_on=True,
            zorder=6,
        )

    # 中文注释：分图编号缩小并贴近各子图左上角，符合期刊主图习惯。
    heat_position = heat_ax.get_position()
    updated_ridge_position = ridge_ax.get_position()
    panel_label_y = heat_position.y1 + 0.008
    fig.text(
        heat_position.x0 - 0.060,
        panel_label_y,
        "a",
        fontsize=24,
        fontweight="bold",
        va="top",
    )
    fig.text(
        updated_ridge_position.x0 - 0.01,
        panel_label_y,
        "b",
        fontsize=24,
        fontweight="bold",
        ha="right",
        va="top",
    )

    fig.savefig(output_path, dpi=1200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[左右双栏主图] {output_path}")

def main() -> None:
    """默认输出左右双栏组合主图。"""
    _configure_chinese_font()
    case_results = _load_case_results()
    if not case_results:
        raise FileNotFoundError("没有找到可用的六案例预测CSV")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(SCI_RC):
        plot_case_studies_bars_ridgeline(case_results)


def plot_case_studies_map_ridgeline(
    case_results: dict[str, pd.DataFrame],
    class_names: list[str],
    output_path: Path,
    **_: object,
) -> None:
    """兼容旧调用：不再绘制地图，只输出六联柱图和年龄山脊图。"""
    del class_names, output_path
    plot_six_case_horizontal_bars(case_results)
    plot_high_arc_ridgeline(case_results)


if __name__ == "__main__":
    main()
