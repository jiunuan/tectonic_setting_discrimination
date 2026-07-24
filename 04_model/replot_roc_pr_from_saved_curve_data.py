"""从已保存的 ROC/PR 曲线数据直接重画对比图。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 中文注释：从当前项目集中路径配置读取 ROC/PR 曲线缓存和输出位置。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ROC_PR_CURVE_DATA_CSV, ROC_PR_FROM_SAVED_PNG


# 中文注释：输入、输出路径均使用完整文件地址，避免路径拼接带来的环境差异。
CURVE_DATA_CSV = str(ROC_PR_CURVE_DATA_CSV)
OUTPUT_PNG = str(ROC_PR_FROM_SAVED_PNG)


def _apply_boxed_panel(ax, linewidth: float = 0.6) -> None:
    """统一为太古代时间演化图的黑色细边框风格。"""
    # 中文注释：四条轴脊保留为黑色 0.6 pt，和太古代时间演化主图保持一致。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(linewidth)
        spine.set_capstyle("butt")
    ax.tick_params(width=linewidth)


def _set_roc_pr_style() -> None:
    """设置 ROC/PR 图的统一字体和边框参数。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.6,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#000000",
    })


def _validate_curve_table(curve_df: pd.DataFrame) -> None:
    """检查曲线数据表是否包含重画所需字段。"""
    required_columns = {
        "Curve", "Model", "x", "y", "AUC", "mAP",
        "Color", "LineWidth", "LineStyle",
    }
    missing_columns = required_columns - set(curve_df.columns)
    if missing_columns:
        raise ValueError(f"曲线数据缺少必要列: {sorted(missing_columns)}")


def plot_from_saved_curve_data() -> None:
    """读取保存的曲线点并重画 ROC/PR 双面板图。"""
    curve_df = pd.read_csv(CURVE_DATA_CSV, encoding="utf-8-sig")
    _validate_curve_table(curve_df)
    _set_roc_pr_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 中文注释：按 CSV 中出现的模型顺序绘制，保持原图例顺序。
    for model_name in curve_df["Model"].drop_duplicates():
        model_df = curve_df[curve_df["Model"] == model_name]
        style_row = model_df.iloc[0]
        color = style_row["Color"]
        linewidth = float(style_row["LineWidth"])
        linestyle = style_row["LineStyle"]
        auc_value = float(style_row["AUC"])
        map_value = float(style_row["mAP"])

        roc_df = model_df[model_df["Curve"] == "ROC"]
        axes[0].plot(
            roc_df["x"].to_numpy(dtype=float),
            roc_df["y"].to_numpy(dtype=float),
            color=color,
            lw=linewidth,
            linestyle=linestyle,
            label=f"{model_name} (AUC = {auc_value:.3f})",
        )

        pr_df = model_df[model_df["Curve"] == "PR"]
        axes[1].plot(
            pr_df["x"].to_numpy(dtype=float),
            pr_df["y"].to_numpy(dtype=float),
            color=color,
            lw=linewidth,
            linestyle=linestyle,
            label=f"{model_name} (mAP = {map_value:.3f})",
        )

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1.0, linestyle="--", zorder=0)
    ax.set_xlim(-0.005, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _apply_boxed_panel(ax)
    leg_a = ax.legend(loc="lower right", edgecolor="#000000",
                      fancybox=False, framealpha=0.95)
    leg_a.get_frame().set_linewidth(0.6)
    # 中文注释：子图编号按要求进一步放大。
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top", ha="left")

    ax = axes[1]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _apply_boxed_panel(ax)
    leg_b = ax.legend(loc="lower left", edgecolor="#000000",
                      fancybox=False, framealpha=0.95)
    leg_b.get_frame().set_linewidth(0.6)
    # 中文注释：子图编号按要求进一步放大。
    ax.text(-0.14, 1.04, "b", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top", ha="left")

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()
    print(f"ROC/PR 重绘完成: {OUTPUT_PNG}")


if __name__ == "__main__":
    plot_from_saved_curve_data()
