from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 太古代时间演化——分箱敏感性分析（独立附图）
#
#   目的：验证时间演化主图中识别出的三个时间特征
#       · ~3.8 Ga      arc-like window
#       · ~3.6–3.3 Ga  Ba-proxy decoupling
#       · ~2.7–2.5 Ga  late Archean rise
#   不是分箱边界造成的人为产物（binning artefact）。
#
#   做法（两条独立检验，互为补充）：
#     (a) 多分箱宽度对照：100 / 150 / 200 / 300 Myr bins，比较 GeoDAN 全样品弧亲和性
#         与 Ba/Th median 的时间曲线是否在不同 bin 宽下保持同一趋势。
#     (b) 200 Myr 起点平移检验：固定 200 Myr 窗宽，把分箱起点平移 0 / 50 / 100 / 150 Ma，
#         若特征位置与方向不随起点漂移，则不是 bin 边界造成的。
#
#   ★ 本脚本独立运行，复用原模块的数据读取与分箱流程，不修改任何原文件。
#   输出：时间分箱敏感性图（≥600 dpi）
# ──────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_PREDICT_DIR = str(Path(__file__).resolve().parent)
for _p in (_PROJECT_ROOT, _PREDICT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import archean_vit_transformer_dualstream_predict_analysis as A  # noqa: E402
import archean_time_evolution as R  # noqa: E402  复用配色/窗口/装饰，保持与主图一致
from config.paths import ZENODO_ARCHEAN_PREDICTIONS_CSV  # noqa: E402


PREDICTION_CSV = Path(ZENODO_ARCHEAN_PREDICTIONS_CSV) if Path(ZENODO_ARCHEAN_PREDICTIONS_CSV).exists() else A.FINAL_PREDICTION_PATH
OUTPUT_PATH = A.FINAL_OUTPUT_DIR / "fig9_sensitivity.png"

# 中文注释：右端留到 2.40 Ga，避免最年轻 bin 的数据点贴边被裁剪（与主图一致）。
X_OLD, X_YOUNG = 4.00, 2.40
X_TICKS = np.arange(2.6, 4.01, 0.2)

BIN_SIZES_MYR = [100, 150, 200, 300]
SHIFTS_MYR = [0, 50, 100, 150]          # 200 Myr 窗宽下的起点平移量
SLIDING_WIDTH_MYR = 200

BINSIZE_COLORS = {100: "#9E9E9E", 150: "#5BA1D0", 200: "#2E5C8A", 300: "#C0392B"}
SHIFT_COLORS = ["#2E5C8A", "#2E8B57", "#E08214", "#8E44AD"]


# ──────────────────────────────────────────────────────────────────────────────
# 数据
# ──────────────────────────────────────────────────────────────────────────────

def load_samples() -> pd.DataFrame:
    """读取逐样品预测结果，准备年龄与 Ba/Th 比值列（复用原数据口径）。"""
    df = A.read_csv_fallback(PREDICTION_CSV)
    age = pd.to_numeric(df["C_AGE"], errors="coerce")
    if "AGE" in df.columns:
        age = age.fillna(pd.to_numeric(df["AGE"], errors="coerce"))
    df["_age_ma"] = age
    df["_arc"] = pd.to_numeric(df["Arc_probability3"], errors="coerce")
    ba = pd.to_numeric(df["BA"], errors="coerce")
    th = pd.to_numeric(df["TH"], errors="coerce")
    df["_ba_th"] = ba / th.replace(0, np.nan)
    return df.dropna(subset=["_age_ma"]).reset_index(drop=True)


def _fixed_bins(df: pd.DataFrame, bin_size: int, shift: int = 0) -> pd.DataFrame:
    """以给定 bin 宽与起点平移量分箱，返回逐箱 (age_mid_ga, mean_arc, median_ba_th, n)。"""
    age = df["_age_ma"].to_numpy(dtype=float)
    start = int(np.floor(age.min() / bin_size) * bin_size) - shift
    stop = int(np.ceil(age.max() / bin_size) * bin_size + bin_size)
    edges = np.arange(start, stop + 1, bin_size)
    cats = pd.cut(age, bins=edges, right=False, include_lowest=True)
    g = df.assign(_bin=cats).dropna(subset=["_bin"]).groupby("_bin", observed=True)
    rows = []
    for interval, grp in g:
        mid = interval.mid / 1000.0
        if mid < 2.5 or mid > 4.0:
            continue
        rows.append({
            "age_mid_ga": mid,
            "mean_arc": float(grp["_arc"].mean()),
            "median_ba_th": float(grp["_ba_th"].median()),
            "n": int(len(grp)),
        })
    return pd.DataFrame(rows).sort_values("age_mid_ga", ascending=False).reset_index(drop=True)


def _sliding_window(df: pd.DataFrame, width: int, shift: int, step: int = 50) -> pd.DataFrame:
    """200 Myr 滑动窗口：窗心从 (2500+shift) 起，按 step 滑动，返回逐窗均值曲线。"""
    age = df["_age_ma"].to_numpy(dtype=float)
    arc = df["_arc"].to_numpy(dtype=float)
    half = width / 2.0
    centers = np.arange(2500 + shift, 4000 + 1, step)
    rows = []
    for c in centers:
        m = (age >= c - half) & (age < c + half)
        if m.sum() < 5:
            continue
        rows.append({"age_mid_ga": c / 1000.0, "mean_arc": float(np.nanmean(arc[m])), "n": int(m.sum())})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────────────────────────────────────

def _decorate(ax, *, show_xlabel: bool) -> None:
    """统一窗口阴影、坐标轴与风格，与主图一致（两套视觉语言：文献背景 vs 数据信号）。"""
    # 外部文献背景带（宽、主导）
    ax.axvspan(R.TRANSITION_CONTEXT["lo"], R.TRANSITION_CONTEXT["hi"],
               color=R.TRANSITION_CONTEXT["color"], alpha=R.WINDOW_ALPHA,
               lw=0.0, zorder=-2)
    # 数据信号①：~3.8 Ga 窄带
    ax.axvspan(R.ARC_NARROW_WINDOW["lo"], R.ARC_NARROW_WINDOW["hi"],
               color=R.ARC_NARROW_WINDOW["color"], alpha=R.NARROW_ALPHA,
               lw=0.0, zorder=0)
    # 数据信号②：~3.5 Ga transient arc pulse 竖向引导点线
    ax.axvline(R.TRANSIENT_PULSE["age"], color=R.TRANSIENT_PULSE["color"],
               linestyle=(0, (1, 2.2)), linewidth=0.9, alpha=0.6, zorder=1)
    ax.set_xlim(X_OLD, X_YOUNG)
    ax.set_xticks(X_TICKS)
    ax.set_facecolor("#FFFFFF")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(R.SPINE_GRAY)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, color=R.GRID_GRAY, alpha=0.5)
    if show_xlabel:
        ax.set_xlabel("Age (Ga)", fontsize=10.5)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def build_sensitivity_figure(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.6, 8.4))
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.16,
                          left=0.10, right=0.975, top=0.95, bottom=0.075)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)

    # ── (a) 多分箱宽度：GeoDAN 全样品弧亲和性 ──────────────────────────────────
    _decorate(ax_a, show_xlabel=False)
    for bs in BIN_SIZES_MYR:
        tab = _fixed_bins(df, bs)
        lw = 2.0 if bs == 200 else 1.3
        ax_a.plot(tab["age_mid_ga"], tab["mean_arc"], color=BINSIZE_COLORS[bs],
                  linewidth=lw, marker="o", markersize=4.2, markerfacecolor="white",
                  markeredgecolor=BINSIZE_COLORS[bs], markeredgewidth=1.1,
                  alpha=0.95, zorder=4, label=f"{bs} Myr bins")
    ax_a.set_ylabel("GeoDAN mean\narc affinity", fontsize=10.5)
    # 中文注释：给图例预留头部空间，避免 100 Myr 曲线峰值顶到图例框。
    _, y_top = ax_a.get_ylim()
    ax_a.set_ylim(top=y_top * 1.18)
    ax_a.legend(loc="upper center", fontsize=7.6, ncol=4, frameon=True,
                framealpha=0.9, edgecolor="0.8", columnspacing=1.0, handlelength=1.8)
    ax_a.text(-0.085, 1.0, "(a)", transform=ax_a.transAxes, fontsize=13,
              fontweight="bold", va="top", ha="left")

    # ── (b) 多分箱宽度：Ba/Th median（独立地化指标交叉验证） ────────────────────
    _decorate(ax_b, show_xlabel=False)
    for bs in BIN_SIZES_MYR:
        tab = _fixed_bins(df, bs)
        lw = 2.0 if bs == 200 else 1.3
        ax_b.plot(tab["age_mid_ga"], tab["median_ba_th"], color=BINSIZE_COLORS[bs],
                  linewidth=lw, marker="s", markersize=4.2, markerfacecolor="white",
                  markeredgecolor=BINSIZE_COLORS[bs], markeredgewidth=1.1,
                  alpha=0.95, zorder=4, label=f"{bs} Myr bins")
    ax_b.set_ylabel("Median Ba/Th", fontsize=10.5)
    ax_b.text(-0.085, 1.0, "(b)", transform=ax_b.transAxes, fontsize=13,
              fontweight="bold", va="top", ha="left")

    # ── (c) 200 Myr 起点平移检验：GeoDAN 全样品弧亲和性 ─────────────────────────
    _decorate(ax_c, show_xlabel=True)
    for shift, color in zip(SHIFTS_MYR, SHIFT_COLORS):
        tab = _fixed_bins(df, 200, shift=shift)
        ax_c.plot(tab["age_mid_ga"], tab["mean_arc"], color=color, linewidth=1.5,
                  marker="o", markersize=4.0, markerfacecolor="white",
                  markeredgecolor=color, markeredgewidth=1.0, alpha=0.9, zorder=4,
                  label=f"start +{shift} Ma")
    ax_c.set_ylabel("GeoDAN mean\narc affinity", fontsize=10.5)
    ax_c.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    # 中文注释：图例带标题占两行，多留一些头部空间。
    _, y_top = ax_c.get_ylim()
    ax_c.set_ylim(top=y_top * 1.28)
    ax_c.legend(loc="upper center", fontsize=7.6, ncol=4, frameon=True,
                framealpha=0.9, edgecolor="0.8", title="200 Myr bins, shifted origin",
                title_fontsize=7.6, columnspacing=1.0, handlelength=1.8)
    ax_c.text(-0.085, 1.0, "(c)", transform=ax_c.transAxes, fontsize=13,
              fontweight="bold", va="top", ha="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[OK] 敏感性分析图已保存: {OUTPUT_PATH}")


def main() -> None:
    print("=" * 78)
    print("太古代时间分箱敏感性分析（100/150/200/300 Myr + 200 Myr 起点平移）")
    print("=" * 78)
    df = load_samples()
    build_sensitivity_figure(df)
    print("完成。既有输出文件未改动。")


if __name__ == "__main__":
    main()
