import argparse
from pathlib import Path
import sys

# 中文注释：箱线图读取当前项目现代训练集/总表，输出到 data/figures。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import TRAIN_RAW_CSV, COMBINED_CSV, ZENODO_MODERN_CSV, FIGURES_DIR

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# 默认展示最新版训练流程中“插补前”的训练集实测数据。
# 箱线图用于描述真实观测分布，不使用随机森林插补值、SMOTE 合成值或归一化值。
DATA_SCOPE = "train"

# 直接写完整文件地址，避免路径逐段拼接。
DATA_PATHS = {
    "train": Path(TRAIN_RAW_CSV),
    # 中文注释：总表优先使用 Zenodo 发布表；若用户已按流程目录复制，则回退到 COMBINED_CSV。
    "full": Path(ZENODO_MODERN_CSV if Path(ZENODO_MODERN_CSV).exists() else COMBINED_CSV),
}

# 不动态拼接输出路径，训练集和总表各自使用固定完整地址。
OUTPUT_PATHS = {
    "train": Path(FIGURES_DIR) / "selected_elements" / "selected_element_boxplots_train_observed.png",
    "full": Path(FIGURES_DIR) / "selected_elements" / "selected_element_boxplots_full_observed.png",
}

# 需要绘制的元素列名及其图中显示名称，按面板 a-f 顺序排列。
ELEMENT_LABELS = {
    "BA(PPM)": "Ba (ppm)",
    "TH(PPM)": "Th (ppm)",
    "NB(PPM)": "Nb (ppm)",
    "LA(PPM)": "La (ppm)",
    "DY(PPM)": "Dy (ppm)",
    "TIO2(WT%)": r"TiO$_2$ (wt.%)",
}

# 各元素的纵轴尺度与刻度设置。
# Ba、Th、Nb、La 分布明显右偏，使用 log 尺度并指定规范刻度；Dy、TiO2 使用线性尺度。
ELEMENT_SCALES = {
    "BA(PPM)": ("log", [1, 10, 100, 1000, 10000]),
    "TH(PPM)": ("log", [0.01, 0.1, 1, 10, 100]),
    "NB(PPM)": ("log", [0.1, 1, 10, 100, 1000]),
    "LA(PPM)": ("log", [1, 10, 80]),
    "DY(PPM)": ("linear", None),
    "TIO2(WT%)": ("linear", None),
}

# 少数超高值只影响显示上界，不删除数据；箱体统计仍基于全部有效实测值。
LOG_Y_LIMIT_QUANTILES = {
    "BA(PPM)": 0.998,
    "TH(PPM)": 0.998,
    "NB(PPM)": 0.998,
    "LA(PPM)": 0.995,
}
LOG_Y_LIMIT_PADDING_DECADES = 0.16

# Dy 和 TiO2 用高分位数收紧展示范围，避免少数极值造成大片空白。
LINEAR_Y_LIMIT_QUANTILES = {
    "DY(PPM)": 0.995,
    "TIO2(WT%)": 0.998,
}
LINEAR_Y_LIMIT_PADDING = 1.02

# 将原始构造环境名称统一转换为论文图中更简洁的缩写。
TECTONIC_ABBR = {
    "Continental arc": "CA",
    "Island arc": "IA",
    "Intra-oceanic arc": "IOA",
    "BACK-ARC_BASIN": "BAB",
    "SPREADING_CENTER": "MOR",
    "OCEANIC PLATEAU": "OP",
    "OCEAN ISLAND": "OI",
    "CONTINENTAL FLOOD BASALT": "CF",
    "CONTINENTAL_RIFT": "CR",
    "CA": "CA",
    "IA": "IA",
    "IOA": "IOA",
    "BAB": "BAB",
    "MOR": "MOR",
    "OP": "OP",
    "OI": "OI",
    "CF": "CF",
    "CFB": "CF",
    "CR": "CR",
}

# 固定横轴类别顺序，便于跨子图直接比较。
TECTONIC_ORDER = ["CA", "IA", "IOA", "BAB", "MOR", "OP", "OI", "CR", "CF"]
CATEGORY_SPACING = 1.20

# 每个构造环境对应一种固定颜色，使用低饱和、分组一致的期刊风格配色。
TECTONIC_COLORS = {
    "CA": "#C9826E",
    "IA": "#D8A06F",
    "IOA": "#D6B67A",
    "BAB": "#8EA7B8",
    "MOR": "#78AAA2",
    "OP": "#8F91BA",
    "OI": "#B2A0C7",
    "CR": "#B7A36F",
    "CF": "#9E9576",
}

# jitter 散点叠加参数：散点只用于展示分布形态，箱体统计始终基于全部数据。
JITTER_WIDTH = 0.18
JITTER_MAX_POINTS = 120
SCATTER_SIZE = 2.4
SCATTER_ALPHA = 0.14
SCATTER_HALO_SIZE = 7.2
SCATTER_HALO_ALPHA = 0.055
BOX_ALPHA = 0.78
BOX_WIDTH = 0.78
JITTER_SEED = 0


def load_dataset(scope: str) -> pd.DataFrame:
    """读取训练集或划分前总表。"""
    return pd.read_csv(DATA_PATHS[scope], low_memory=False)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """检查必要列、统一构造环境标签，并将元素列转换为数值型。"""
    required_columns = ["TECTONIC SETTING", *ELEMENT_LABELS.keys()]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 只保留本图需要的列，避免无关列影响后续统计。
    plot_df = df[required_columns].copy()
    plot_df["TECTONIC SETTING"] = plot_df["TECTONIC SETTING"].map(TECTONIC_ABBR)
    plot_df = plot_df.dropna(subset=["TECTONIC SETTING"])

    for element in ELEMENT_LABELS:
        # 异常字符串转为 NaN，绘图时会自动忽略这些缺失值。
        plot_df[element] = pd.to_numeric(plot_df[element], errors="coerce")

    return plot_df


def print_dataset_summary(df: pd.DataFrame, scope: str) -> None:
    """在终端输出样本量、类别数量和绘图元素缺失值，方便作图核对。"""
    print(f"Data scope: {scope}")
    print("Value source: observed values before imputation")
    print(f"Rows: {len(df)}")
    print("\nClass counts:")
    print(df["TECTONIC SETTING"].value_counts().reindex(TECTONIC_ORDER).to_string())
    print("\nMissing values in plotted elements:")
    print(df[list(ELEMENT_LABELS.keys())].isna().sum().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按构造环境绘制 Ba、Th、Nb、La、Dy、TiO2 箱型图。"
    )
    parser.add_argument(
        "--scope",
        choices=["train", "full"],
        default=DATA_SCOPE,
        help=(
            "train：最新版训练流程中插补前的训练集实测值；"
            "full：划分训练集/测试集之前的完整实测总表。"
        ),
    )
    return parser.parse_args()


def _format_log_tick(value: float) -> str:
    """log 轴刻度标签：使用千分位，避免科学计数法。"""
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def _format_linear_tick(value: float, _pos: int) -> str:
    """线性轴刻度标签：整数直接显示，必要时保留最少小数位。"""
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _style_axis(ax: plt.Axes) -> None:
    """统一子图外观：四边框、细坐标轴、淡水平网格、白底。"""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.6)

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        which="major",
        linestyle=(0, (1, 3)),
        linewidth=0.4,
        color="#d6d6d6",
        alpha=0.65,
    )
    ax.grid(axis="x", visible=False)

    # 中文注释：整体字体略微放大，刻度从 8 pt 调整到 9 pt。
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=3, colors="#333333")
    ax.tick_params(axis="both", which="minor", length=0)
    for label in ax.get_yticklabels():
        label.set_fontweight("semibold")
    ax.set_xlabel("")
    ax.set_ylabel("")


def _panel_tag(ax: plt.Axes, tag: str) -> None:
    """子图左上角外侧使用无括号、加粗的大号编号，风格对齐 PCA 图。"""
    # 中文注释：用固定点数偏移定位编号，避免不同子图尺寸下相对位置漂移。
    ax.annotate(
        tag,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-36, -8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="#222222",
        clip_on=False,
        zorder=5,
    )


def create_selected_element_boxplots(df: pd.DataFrame, output_path: Path) -> Path:
    """生成 2×3 的组合箱线图（叠加 jitter 散点），保存为高分辨率 PNG。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # 中文注释：整体字号略微上调，使六联箱型图与 PCA 图的视觉权重更一致。
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    panel_letters = "abcdef"
    n_categories = len(TECTONIC_ORDER)
    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.1), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.03)

    for idx, (ax, (column, name)) in enumerate(zip(axes.flat, ELEMENT_LABELS.items())):
        scale, ticks = ELEMENT_SCALES[column]
        positions: list[float] = []
        box_data: list[np.ndarray] = []
        box_colors: list[str] = []

        for category_index, category in enumerate(TECTONIC_ORDER, start=1):
            pos = category_index * CATEGORY_SPACING
            values = df.loc[df["TECTONIC SETTING"] == category, column].dropna()
            if scale == "log":
                # 中文注释：log 尺度无法显示非正值，先剔除 <=0 的数据。
                values = values[values > 0]
            array = values.to_numpy()
            if array.size == 0:
                continue

            color = TECTONIC_COLORS[category]
            positions.append(pos)
            box_data.append(array)
            box_colors.append(color)

            # 中文注释：大类别只抽样显示散点，避免过度遮挡；箱体仍使用全部数据。
            if array.size > JITTER_MAX_POINTS:
                sample = array[rng.choice(array.size, JITTER_MAX_POINTS, replace=False)]
            else:
                sample = array
            jitter_x = pos + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=sample.size)
            ax.scatter(
                jitter_x,
                sample,
                s=SCATTER_HALO_SIZE,
                color=color,
                alpha=SCATTER_HALO_ALPHA,
                linewidths=0,
                zorder=0.8,
                rasterized=True,
            )
            ax.scatter(
                jitter_x,
                sample,
                s=SCATTER_SIZE,
                color=color,
                alpha=SCATTER_ALPHA,
                linewidths=0,
                zorder=1,
                rasterized=True,
            )

        box = ax.boxplot(
            box_data,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            # 中文注释：须线使用第 2 和第 98 百分位，而不是默认 1.5×IQR。
            whis=(2, 98),
            boxprops=dict(linewidth=0.8, edgecolor="#555555"),
            whiskerprops=dict(linewidth=0.8, color="#555555"),
            capprops=dict(linewidth=0.8, color="#555555"),
            medianprops=dict(linewidth=1.2, color="#222222"),
            zorder=2,
        )
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(BOX_ALPHA)

        all_values = np.concatenate(box_data) if box_data else np.array([1.0])
        if scale == "log":
            ax.set_yscale("log")
            whisker_ydata = np.concatenate([w.get_ydata() for w in box["whiskers"]])
            low = whisker_ydata[whisker_ydata > 0].min()
            if column in LOG_Y_LIMIT_QUANTILES:
                high = np.nanquantile(all_values, LOG_Y_LIMIT_QUANTILES[column])
                high *= 10 ** LOG_Y_LIMIT_PADDING_DECADES
            else:
                high = all_values.max() * 1.12
            ax.set_ylim(low / 1.15, high)
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: _format_log_tick(value)))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        else:
            if column in LINEAR_Y_LIMIT_QUANTILES:
                upper_value = np.nanquantile(all_values, LINEAR_Y_LIMIT_QUANTILES[column])
            else:
                upper_value = all_values.max()
            ax.set_ylim(0, upper_value * LINEAR_Y_LIMIT_PADDING)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_linear_tick))

        category_positions = [index * CATEGORY_SPACING for index in range(1, n_categories + 1)]
        ax.set_xticks(category_positions)
        ax.set_xticklabels(
            TECTONIC_ORDER,
            fontfamily="Microsoft YaHei",
            fontweight="semibold",
            fontsize=7,
            ha="center",
        )
        if idx < 3:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlim(0.5 * CATEGORY_SPACING, (n_categories + 0.5) * CATEGORY_SPACING)

        _style_axis(ax)

        ylabel_pad = 0 if idx in (0, 3) else 3
        ax.set_ylabel(name, fontsize=11, color="#333333", fontweight="semibold", labelpad=ylabel_pad)

        # 中文注释：编号样式模仿 PCA 图，使用无括号粗体小写字母。
        _panel_tag(ax, panel_letters[idx])

    for column_axes in axes.T:
        fig.align_ylabels(column_axes)

    fig.supxlabel("Tectonic setting", fontsize=13, color="#222222", fontweight="semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    # 命令示例：
    # python data_analysis/selected_element_boxplots.py --scope train
    # python data_analysis/selected_element_boxplots.py --scope full
    args = parse_args()
    dataset = load_dataset(args.scope)
    dataset = prepare_dataset(dataset)
    print_dataset_summary(dataset, args.scope)
    output_path = create_selected_element_boxplots(dataset, OUTPUT_PATHS[args.scope])
    print(f"\nBoxplot saved to: {output_path}")
