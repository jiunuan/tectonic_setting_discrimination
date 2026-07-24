from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 太古代时间演化图——共享时间轴下的多证据对齐图
#
#   本脚本是 archean_vit_transformer_dualstream_predict_analysis.py 中
#   plot_main_composite_figure 的独立重构版本。
#
#   设计目标：把结果堆叠图改为"共享时间轴下的多证据对齐图"，
#   风格参考 Nature Communications / Nature Geoscience / Earth-Science Reviews
#   常见的深时演化综合图。图内只保留必要的面板标签、坐标轴、图例、少量时间窗口
#   短标签与误差说明，解释性文字一律留给图注和正文。
#
#   ★ 重要约定：
#     · 本脚本不修改、不调用原绘图函数 plot_main_composite_figure，原脚本与其
#       输出文件（fig_main_composite_tectonic.png）保持不变，可随时回退对比。
#     · 数据处理流程完全复用原脚本：直接 import 原模块的纯数据函数与常量，
#       仅重构图件布局和视觉表达。
#     · 预测推理结果直接读取已保存的 expanded_archean_predictions.csv，因此本图
#       脚本不重新跑模型推理，可独立、快速、可重复地重绘。
#
#   图形语法（本轮重构）：
#     · Panel (a)：九类构造环境填充式 KDE 山脊图，所有类别共用全局高度尺度，
#                  不做类别内归一化，直接比较整体丰度及其随时间的变化；
#     · Panel (b)：log-scale 样本数柱状图（左轴，total + CA+IA+IOA）
#                  + 典型弧端元比例折线（右轴，GeoDAN arc fraction + Liu 参照）；
#     · Panels (c)–(e)：Ba/La、Th/Nb、Nb/La 分别使用独立线性 y 轴绘制
#                      Arc_probability3 加权均值，误差棒为非对称 bootstrap 95% CI。
#
#   输出（均为新文件名，不覆盖原图）：
#     · fig9_redesign_main.png            主图（≥600 dpi 位图）
#     · fig9_redesign_panelC_raw.png      不确定性附图：加权均值 + log 轴 + bootstrap 95% CI
#     · fig9_redesign_caption.txt         建议图注（供正文/图注使用，不画进图里）
# ──────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# ── 复用原分析模块的数据处理流程与常量（单一数据真相来源） ──────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_PREDICT_DIR = str(Path(__file__).resolve().parent)
for _p in (_PROJECT_ROOT, _PREDICT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import archean_vit_transformer_dualstream_predict_analysis as A  # noqa: E402
from config.paths import ZENODO_ARCHEAN_PREDICTIONS_CSV  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════════════════════════════════════

# 数据来源：直接复用原主线流程的产物与配置
PREDICTION_CSV = Path(ZENODO_ARCHEAN_PREDICTIONS_CSV) if Path(ZENODO_ARCHEAN_PREDICTIONS_CSV).exists() else A.FINAL_PREDICTION_PATH  # 已保存的 9 分类预测结果
SOURCE_S3_CSV  = A.SOURCE_S3_CSV_PATH             # Liu et al. 2024 原始数据（外部参照）
TRAIN_CSV      = A.TRAIN_PATH                     # 读取类别顺序
BIN_SIZE_MYR   = A.BIN_SIZE_MYR                   # 主图分箱（200 Myr）
HIGH_STD       = A.HIGH_STD
ARC_THRESHOLD  = 0.5                              # GeoDAN 弧亲和性阈值（与原主图一致）

# 输出目录沿用原最终输出目录，但使用独立文件名，避免覆盖既有结果。
OUTPUT_DIR        = A.FINAL_OUTPUT_DIR
FIG_MAIN_PATH     = OUTPUT_DIR / "fig9_redesign_main.png"
FIG_PANELC_RAW    = OUTPUT_DIR / "fig9_redesign_panelC_raw.png"
CAPTION_TXT_PATH  = OUTPUT_DIR / "fig9_redesign_caption.txt"
INDICATOR_AUDIT_CSV = OUTPUT_DIR / "fig9_redesign_indicator_subset_audit.csv"

# bootstrap 设置
N_BOOT    = 2000
BOOT_SEED = 20240611

# 中文注释：默认按每个比值所需的两个元素做成对有效统计，以保留更多太古代样品。
# 如需严格共同样品敏感性分析，可临时改为 True；正式图不采用该口径。
INDICATOR_REQUIRE_COMPLETE_QUARTET = False
# 共享时间轴范围（老→新，左→右）
# 中文注释：右端留到 2.40 Ga，避免 2.5 Ga bin 的数据点 / 样品数柱被轴边界裁剪。
X_OLD, X_YOUNG = 4.00, 2.40
X_TICKS = np.arange(2.6, 4.01, 0.2)


# ── Panel (a) 九类构造亲和性配色（按地质意义重新分组） ──────────────────────────
#   · CA / IA / IOA：暖色系，表示典型弧端元
#   · BAB         ：青蓝/蓝绿，表示过渡端元
#   · OI / OP / CF：紫色系，表示板内 / 地幔柱相关
#   · MOR         ：冷蓝
#   · CR          ：低饱和粉紫 / 灰粉
REDESIGN_CLASS_COLORS = {
    "Continental arc":          "#B2182B",  # CA  深暖红
    "Island arc":               "#E6783C",  # IA  橙
    "Intra-oceanic arc":        "#F4A85C",  # IOA 琥珀
    "BACK-ARC_BASIN":           "#1B9E91",  # BAB 青蓝/蓝绿（过渡端元）
    "OCEAN ISLAND":             "#6A51A3",  # OI  紫（板内/地幔柱）
    "OCEANIC PLATEAU":          "#9E79C0",  # OP  浅紫
    "CONTINENTAL FLOOD BASALT": "#C0A4D8",  # CF  淡紫
    "SPREADING_CENTER":         "#2C7FB8",  # MOR 冷蓝
    "CONTINENTAL_RIFT":         "#C9A9C0",  # CR  低饱和灰粉
}

# 山脊图行序（自上而下按地质分组排列，使分组结构一目了然）
#   CA / IA / IOA 典型弧端元在最上，BAB 过渡端元紧随，再到板内/地幔柱、MOR、CR。
REDESIGN_ABBREV_ORDER = ["CA", "IA", "IOA", "BAB", "OP", "OI", "CF", "MOR", "CR"]

# ── 太古代地质时代条配色（低饱和淡蓝/淡粉/淡橙/淡绿） ──────────────────────────
PERIOD_BANDS = [
    {"name": "Eoarchean",    "lo": 3.60, "hi": 4.00, "color": "#C8D9EC"},  # 淡蓝
    {"name": "Paleoarchean", "lo": 3.20, "hi": 3.60, "color": "#F0CADA"},  # 淡粉
    {"name": "Mesoarchean",  "lo": 2.80, "hi": 3.20, "color": "#FAE5C2"},  # 淡橙
    {"name": "Neoarchean",   "lo": 2.50, "hi": 2.80, "color": "#CDE8C7"},  # 淡绿
]

# ── 视觉语言：外部文献背景 vs 本文数据信号（两套表达，严格区分）──────────────
# 【外部文献框架】3.2–2.5 Ga 仅用年代条下方的浅橙细线表示，不覆盖数据面板。
TRANSITION_CONTEXT = {
    "lo": 2.50, "hi": 3.20, "color": "#E7B875",
    "label": "3.2-2.5 Ga transition context",
}

# 【本文数据信号①】~3.8 Ga 窄阴影带——仅见于格陵兰、Pilbara 等少数地体的弱弧信号。
#   引 Polat & Hofmann, 2003；Furnes et al., 2009。
ARC_NARROW_WINDOW = {
    "lo": 3.76, "hi": 3.84, "color": "#5B8FBF",
    "label": "~3.8 Ga arc-like\n(few cratons)",
    "label_color": "#2C6FB0",
}
NARROW_ALPHA = 0.32

# 【本文数据信号②】~2.7–2.5 Ga 弧亲和性回升——位于文献转型背景的尾部。
LATE_ARCHEAN_RISE = {
    "lo": 2.50, "hi": 2.70, "color": "#8FCB86",
    "label": "~2.7–2.5 Ga\narc-affinity rise",
}
LATE_RISE_ALPHA = 0.34

# 【本文数据信号③】~3.5 Ga transient arc pulse——点状标注，由 (b) 面板 GeoDAN 与
#   Liu et al. (2024) 两条独立曲线同步验证；不配文献式阴影。
TRANSIENT_PULSE = {
    "age": 3.50, "color": "#1F4E79",
    "label": "~3.5 Ga transient\n arc-affinity pulse",
}

# ── 曲线配色 ────────────────────────────────────────────────────────────────
COL_GEODAN_ALL  = "#2E5C8A"   # GeoDAN all samples（蓝）
COL_GEODAN_HIGH = "#C0392B"   # GeoDAN high-confidence（红）
COL_LIU         = "#8C8C8C"   # Liu et al., 2024 外部参照（灰）

COL_BA_LA = "#2E8B57"   # Ba/La 绿
COL_TH_NB = "#2C7FB8"   # Th/Nb 蓝
COL_NB_LA = "#C0392B"   # Nb/La 红

# ── Panel (b) 样本数柱状图配色 ───────────────────────────────────────────────
COL_BAR_TOTAL = "#9E9E9E"   # 全部太古代玄武岩样品（灰）
COL_BAR_ARC   = "#C0392B"   # GeoDAN 典型弧端元 CA+IA+IOA（红）
COL_LINE_ARC  = "#2E5C8A"   # GeoDAN 典型弧端元比例折线（蓝）

GRID_GRAY = "#C8C8C8"
SPINE_GRAY = "#000000"
BINLINE_GRAY = "#E6E6E6"    # 200 Myr bin centers 竖向虚线（很浅）
WINDOW_ALPHA = 0.25         # 增强缓冲区颜色，仍保持数据曲线可辨识


# ══════════════════════════════════════════════════════════════════════════════
#  数据装载与统计（复用原模块的数据处理流程）
# ══════════════════════════════════════════════════════════════════════════════

def load_core_frames() -> dict:
    """
    复用原模块的数据处理流程，得到主图所需的全部统计表：
      · df               ：逐样品预测结果（含 age_bin）
      · class_names      ：9 类名称顺序
      · age_summary      ：按年龄分箱统计（含 mean_arc_probability、n_all）
      · thr_summary      ：GeoDAN 弧亲和性 ≥ 阈值的逐箱比例
      · liu_summary      ：Liu et al. 2024 外部参照逐箱比例
    """
    df = A.read_csv_fallback(PREDICTION_CSV)
    # 中文注释：仅排除 Ba≈23349 ppm 的明确异常记录，不设置通用 Ba 阈值。
    # 同时使用样品编号和 Ba 浓度定位，避免误删其他高 Ba 样品。
    ba = pd.to_numeric(df["BA"], errors="coerce")
    sample_id = df["SAMPLE_ID"].astype(str)
    ba_23349_outlier = sample_id.eq("s_2C-14 [24900]") & (ba > 23000.0)
    n_removed = int(ba_23349_outlier.sum())
    if n_removed:
        print(f"[GeoDAN数据质量] 剔除 Ba≈23349 ppm 异常样品: {n_removed}")
    df = df.loc[~ba_23349_outlier].copy()
    df = A.add_age_bins(df, BIN_SIZE_MYR)
    class_names = A.load_class_names(TRAIN_CSV)

    age_summary = A.summarize_by_age(df, class_names)
    # 中文注释：_compute_arc_ratio_by_threshold 仅依赖 df['Arc_probability3']，
    # probs_mean 参数在函数体内并未使用，这里传占位数组即可。
    thr_summary = A._compute_arc_ratio_by_threshold(
        df, np.zeros((len(df), 1)), ARC_THRESHOLD, HIGH_STD
    )
    liu_summary = A.summarize_liu_baseline_by_age(SOURCE_S3_CSV, BIN_SIZE_MYR)

    return {
        "df": df,
        "class_names": class_names,
        "age_summary": age_summary,
        "thr_summary": thr_summary,
        "liu_summary": liu_summary,
    }


def _weighted_bootstrap_mean_ci(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int, float, float]:
    """计算加权均值、bootstrap SEM及非对称95% CI。"""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x = values[valid]
    w = weights[valid]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan, n, float(np.sum(w)), np.nan)

    weighted_mean = float(np.average(x, weights=w))
    idx = rng.integers(0, n, size=(n_boot, n))
    sampled_values = x[idx]
    sampled_weights = w[idx]
    boot_means = np.sum(sampled_values * sampled_weights, axis=1) / np.sum(
        sampled_weights, axis=1
    )
    sem = float(np.std(boot_means, ddof=1))
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    weight_sum = float(np.sum(w))
    effective_n = float(weight_sum ** 2 / np.sum(w ** 2))
    return (
        weighted_mean, sem, float(ci_lo), float(ci_hi),
        n, weight_sum, effective_n,
    )


def compute_indicator_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    计算 Ba/La、Th/Nb、Nb/La 三项独立地球化学指标的逐箱统计：
      · 逐样品比值 → 逐 200 Myr 箱 Arc_probability3 加权均值
      · 对样品行做有放回 bootstrap，每次重算加权均值
      · 误差棒使用 bootstrap 加权均值分布的 2.5% 和 97.5% 分位数
      · robust z-score（原始量纲，列 z/z_lo/z_hi，保留供诊断分析）
      · robust z-score（log10 量纲，列 z_log，保留供诊断分析）：
        比值是乘性量纲，先取 log10 再做 robust z-score，
        使 Ba/La、Th/Nb、Nb/La 不同量纲可用于补充诊断分析。

    返回 {metric: DataFrame[age_mid_ga, weighted_mean, bootstrap_sem,
                            bootstrap_ci_lo, bootstrap_ci_hi, n,
                            weight_sum, effective_n, z, z_lo, z_hi, z_log]}
    """
    work = df.copy()
    for col in ["BA", "TH", "NB", "LA"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        # 中文注释：正式预处理将非正值视为缺失；比值统计必须沿用相同口径，
        # 避免把未检出或缺失编码的 0 错当成真实地球化学比值。
        work[col] = work[col].where(work[col] > 0)
    work["Ba_La"] = work["BA"] / work["LA"].replace(0, np.nan)
    work["Th_Nb"] = work["TH"] / work["NB"].replace(0, np.nan)
    work["Nb_La"] = work["NB"] / work["LA"].replace(0, np.nan)
    work["indicator_weight"] = pd.to_numeric(
        work["Arc_probability3"], errors="coerce"
    ).clip(lower=0.0)
    if INDICATOR_REQUIRE_COMPLETE_QUARTET:
        # 中文注释：该开关仅用于四元素共同完整样品的敏感性分析。
        work = work.dropna(subset=["BA", "LA", "TH", "NB"]).copy()

    rng = np.random.default_rng(BOOT_SEED)
    metrics = ["Ba_La", "Th_Nb", "Nb_La"]
    out: dict[str, list[dict]] = {m: [] for m in metrics}

    for interval, group in work.dropna(subset=["age_bin"]).groupby("age_bin", observed=True):
        mid = interval.mid / 1000.0
        if mid < 2.5:
            continue
        for m in metrics:
            mean, sem, ci_lo, ci_hi, n, weight_sum, effective_n = (
                _weighted_bootstrap_mean_ci(
                group[m].to_numpy(dtype=float),
                group["indicator_weight"].to_numpy(dtype=float),
                N_BOOT,
                rng,
                )
            )
            out[m].append({
                "age_mid_ga": mid,
                "weighted_mean": mean,
                "bootstrap_sem": sem,
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "n": n,
                "weight_sum": weight_sum,
                "effective_n": effective_n,
            })

    def _robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        center = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - center))
        scale = 1.4826 * mad if mad > 0 else (np.nanstd(values) or 1.0)
        return (values - center) / scale, center, scale

    result: dict[str, pd.DataFrame] = {}
    for m in metrics:
        tab = pd.DataFrame(out[m]).sort_values("age_mid_ga", ascending=False).reset_index(drop=True)
        # robust z-score（原始量纲，保留供诊断分析）
        mean_series = tab["weighted_mean"].to_numpy(dtype=float)
        z, center, scale = _robust_z(mean_series)
        tab["z"]    = z
        tab["z_lo"] = (tab["bootstrap_ci_lo"].to_numpy(dtype=float) - center) / scale
        tab["z_hi"] = (tab["bootstrap_ci_hi"].to_numpy(dtype=float) - center) / scale
        # robust z-score（log10 量纲，保留供诊断分析）
        log_mean = np.log10(mean_series)
        tab["z_log"], _, _ = _robust_z(log_mean)
        result[m] = tab
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  共享绘图装饰
# ══════════════════════════════════════════════════════════════════════════════

def _bin_center_ages(age_summary: pd.DataFrame) -> np.ndarray:
    """提取可见区间（≥2.5 Ga）内的 200 Myr bin 中心年龄。"""
    mids = age_summary["age_mid_ga"].to_numpy(dtype=float)
    return np.sort(mids[mids >= 2.5])


def _draw_bin_center_lines(ax, centers: np.ndarray) -> None:
    """在面板内竖向虚线标出 200 Myr bin centers（很浅，不抢数据）。"""
    for c in centers:
        ax.axvline(c, color=BINLINE_GRAY, linestyle=(0, (2, 3)),
                   linewidth=0.55, zorder=0.5)


def _draw_windows(ax) -> None:
    """绘制本文数据窗口；外部文献背景仅在年代条下方用细线表示。"""
    # 【本文数据信号①】~3.8 Ga 窄阴影带（蓝，少数地体弱弧信号）。
    ax.axvspan(ARC_NARROW_WINDOW["lo"], ARC_NARROW_WINDOW["hi"],
               color=ARC_NARROW_WINDOW["color"], alpha=NARROW_ALPHA,
               linewidth=0.0, zorder=0)
    # 【本文数据信号②】晚太古代弧亲和性回升，视觉权重高于顶部文献背景细线。
    ax.axvspan(LATE_ARCHEAN_RISE["lo"], LATE_ARCHEAN_RISE["hi"],
               color=LATE_ARCHEAN_RISE["color"], alpha=LATE_RISE_ALPHA,
               linewidth=0.0, zorder=0)
    # 中文注释：3.5 Ga 转折引导线使用较深、较粗的长虚线，使其贯穿三个面板时更醒目。
    ax.axvline(TRANSIENT_PULSE["age"], color=TRANSIENT_PULSE["color"],
               linestyle=(0, (3.2, 2.2)), linewidth=1.35, alpha=0.90, zorder=2)


def _annotate_events_in_panel_a(ax) -> None:
    """Panel (a) 顶部仅标注本文数据识别出的具体信号。"""
    transform = ax.get_xaxis_transform()

    # ── 本文数据信号①：~3.8 Ga 窄带标签（黑色、粗体；顶端对齐 0.985） ──
    arc_mid = (ARC_NARROW_WINDOW["lo"] + ARC_NARROW_WINDOW["hi"]) / 2.0
    ax.text(arc_mid, 0.955, ARC_NARROW_WINDOW["label"],
            transform=transform, ha="center", va="top",
            fontsize=8, fontweight="bold", color="#000000",
            linespacing=0.95, zorder=7)

    # ── 本文数据信号②：~3.5 Ga transient arc pulse ──
    # 中文注释：文字顶端对齐到 0.985（与其余标签齐平），倒三角 marker 移到文字下方，
    # 既消除"文字被三角顶下去"的错位，又让三角朝下指向 (b) 双曲线峰的引导线。
    ax.text(TRANSIENT_PULSE["age"] - 0.035, 0.955, TRANSIENT_PULSE["label"],
            transform=transform, ha="left", va="top",
            fontsize=8, fontweight="bold", color="#000000",
            linespacing=0.95, zorder=8)
    ax.plot([TRANSIENT_PULSE["age"]], [0.865], transform=transform, marker="v",
            markersize=7.2, markerfacecolor=TRANSIENT_PULSE["color"],
            markeredgecolor="white", markeredgewidth=0.7, clip_on=False, zorder=8)

    # ── 本文数据信号③：文献大背景尾部的数据观测回升 ──
    late_mid = (LATE_ARCHEAN_RISE["lo"] + LATE_ARCHEAN_RISE["hi"]) / 2.0
    ax.text(late_mid, 0.955, LATE_ARCHEAN_RISE["label"],
            transform=transform, ha="center", va="top",
            fontsize=7.5, fontweight="bold", color="#276D3D",
            linespacing=0.95, zorder=7)


def _apply_boxed_panel(ax, linewidth: float = 0.6) -> None:
    """给坐标轴加四条边框边界线。"""
    # 中文注释：所有面板保留上、下、左、右四条轴脊，形成完整图框。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_GRAY)
        spine.set_linewidth(linewidth)
        spine.set_capstyle("butt")


def _style_panel(ax, *, show_xlabel: bool) -> None:
    """统一面板风格：白底、浅灰四边框、反向时间轴。"""
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(X_OLD, X_YOUNG)
    ax.set_xticks(X_TICKS)
    # 中文注释：所有主面板的 y 轴刻度标签统一为 9 pt。
    ax.tick_params(axis="y", labelsize=9)
    _apply_boxed_panel(ax)
    if show_xlabel:
        ax.set_xlabel("Age (Ga)", fontsize=10.5)
    else:
        # 中文注释：中间面板只保留底部轴线作为面板分隔，不显示重复的刻度线和标签。
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


# ══════════════════════════════════════════════════════════════════════════════
#  顶部：仅保留与 Panel (a) 紧密贴合的地质时代条
# ══════════════════════════════════════════════════════════════════════════════

def _draw_top_strip(ax_top) -> None:
    ax_top.set_xlim(X_OLD, X_YOUNG)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    # 中文注释：年代矩形位于上部；下部仅放外部文献背景细线和小号 context 文字。
    # 整个装饰轴仍与 Panel (a) 零间距相接，不产生空白缝隙。
    # 中文注释：年代条底边稍微下移，同时增加高度，使顶边仍贴紧绘图区而不留白缝。
    band_y0, band_h = 0.30, 0.60
    for p in PERIOD_BANDS:
        lo, hi = min(p["lo"], p["hi"]), max(p["lo"], p["hi"])
        ax_top.add_patch(mpatches.Rectangle(
            (lo, band_y0), hi - lo, band_h, facecolor=p["color"],
            edgecolor="white", linewidth=0.8, alpha=0.82, zorder=2,
        ))
        ax_top.text((lo + hi) / 2.0, band_y0 + band_h / 2.0, p["name"],
                    ha="center", va="center", fontsize=8.0, fontweight="bold",
                    color="#33373B", zorder=3)

    # 中文注释：文字嵌在 bracket 横线中间，白底仅遮断线条，不绘制包围文字的边框。
    context_mid = (TRANSITION_CONTEXT["lo"] + TRANSITION_CONTEXT["hi"]) / 2.0
    ax_top.plot(
        [TRANSITION_CONTEXT["lo"], TRANSITION_CONTEXT["hi"]],
        [0.13, 0.13],
        color=TRANSITION_CONTEXT["color"], linewidth=1.25,
        solid_capstyle="butt", zorder=3,
    )
    for x_end in (TRANSITION_CONTEXT["lo"], TRANSITION_CONTEXT["hi"]):
        # 中文注释：加长并加粗两端竖帽，使 bracket 根部在高分辨率图中更清楚。
        ax_top.plot(
            [x_end, x_end], [0.0, 0.26],
            color=TRANSITION_CONTEXT["color"], linewidth=1.25,
            solid_capstyle="butt", zorder=3,
        )
    ax_top.text(
        context_mid + 0.08, 0.13, TRANSITION_CONTEXT["label"],
        ha="center", va="center", fontsize=8.0,
        color="#8A6332", zorder=4,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Panel (a)：九类构造环境的全局尺度填充式 KDE 山脊图
# ══════════════════════════════════════════════════════════════════════════════

def _draw_panel_a(ax, df: pd.DataFrame, class_names: list[str], centers: np.ndarray) -> None:
    counts = (
        df.dropna(subset=["age_bin"])
        .groupby(["age_bin_mid_ma", "pred_class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for name in class_names:
        if name not in counts:
            counts[name] = 0

    class_by_abbrev = {A.CLASS_ABBREVS.get(n, n): n for n in class_names}
    ordered = [class_by_abbrev[ab] for ab in REDESIGN_ABBREV_ORDER if ab in class_by_abbrev]
    counts = counts[ordered].sort_index(ascending=True)
    totals = counts.sum(axis=1).replace(0, np.nan)
    ratios = counts.div(totals, axis=0).fillna(0.0)
    age_ga = ratios.index.to_numpy(dtype=float) / 1000.0
    grid = np.linspace(X_YOUNG, X_OLD, 520)

    # 中文注释：每个年龄箱均以该箱全部样本为统一分母，随后对年龄序列插值和平滑。
    # 九类构造环境不分别归一化，而是共用全局最大值确定山脊高度比例。
    profiles = {
        cls: A._smooth_ridge_curve(
            np.interp(
                grid,
                age_ga,
                ratios[cls].to_numpy(dtype=float),
                left=0.0,
                right=0.0,
            ),
            sigma=12.0,
        )
        for cls in ordered
    }
    global_peak = max((float(profile.max()) for profile in profiles.values()), default=0.0)
    height_scale = 0.78 / global_peak if global_peak > 0 else 1.0
    n_cls = len(ordered)

    _draw_windows(ax)
    _draw_bin_center_lines(ax, centers)
    _annotate_events_in_panel_a(ax)

    # 中文注释：使用半透明面积形成 KDE 山脊，而不是仅绘制折线。
    # 轮廓线只用于界定填充边缘，山脊实际高度仍保留类别间的全局可比性。
    for row_idx, cls in enumerate(ordered):
        baseline = n_cls - 1 - row_idx
        color = REDESIGN_CLASS_COLORS.get(cls, "#999999")
        ridge = profiles[cls] * height_scale
        ax.hlines(
            baseline, X_YOUNG, X_OLD,
            color="#B8B8B8", linewidth=0.55, zorder=2,
        )
        ax.fill_between(
            grid, baseline, baseline + ridge,
            color=color, alpha=0.58, linewidth=0.0, zorder=3,
        )
        ax.plot(
            grid, baseline + ridge,
            color=color, linewidth=1.05, alpha=0.95, zorder=4,
        )

    # 中文注释：顶部留出一行空间放缓冲区内的事件标注。
    ax.set_ylim(-0.25, n_cls + 1.0)
    ax.set_yticks([n_cls - 1 - i + 0.18 for i in range(n_cls)])
    ax.set_yticklabels([A.CLASS_ABBREVS.get(cls, cls) for cls in ordered], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_ylabel("Tectonic setting", fontsize=10.5)
    ax.grid(False)

# ══════════════════════════════════════════════════════════════════════════════
#  弧比例 + 95% bootstrap CI（本文数据信号用实心曲线 + 置信区间呈现）
# ══════════════════════════════════════════════════════════════════════════════

def _proportion_ci(indicator: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    """对 0/1 指示数组做 bootstrap，返回 (比例, ci_lo, ci_hi)。"""
    ind = indicator[np.isfinite(indicator)]
    n = len(ind)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    frac = float(ind.mean())
    # if n < 5:
    #     return (frac, frac, frac)
    boots = ind[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (frac, float(lo), float(hi))


def _geodan_arc_fraction_ci(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """逐 200 Myr 箱计算 GeoDAN Arc_probability3 ≥ 阈值 的比例 + 95% bootstrap CI。"""
    rng = np.random.default_rng(BOOT_SEED)
    arc_prob = pd.to_numeric(df["Arc_probability3"], errors="coerce")
    rows = []
    for interval, g in df.dropna(subset=["age_bin"]).groupby("age_bin", observed=True):
        mid = interval.mid / 1000.0
        if mid < 2.5:
            continue
        ind = (arc_prob.loc[g.index] >= threshold).to_numpy(dtype=float)
        frac, lo, hi = _proportion_ci(ind, N_BOOT, rng)
        rows.append({"age_mid_ga": mid, "frac": frac, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).sort_values("age_mid_ga")


def _liu_arc_fraction_ci(liu_summary: pd.DataFrame) -> pd.DataFrame:
    """由 Liu 逐箱计数 (n_liu_samples, n_liu_arc) 重构 0/1 指示数组做 bootstrap CI。"""
    rng = np.random.default_rng(BOOT_SEED + 7)
    rows = []
    for _, r in liu_summary.iterrows():
        mid = float(r["age_mid_ga"])
        if mid < 2.5:
            continue
        n = int(r["n_liu_samples"]); k = int(r["n_liu_arc"])
        if n == 0:
            continue
        ind = np.concatenate([np.ones(k), np.zeros(n - k)])
        frac, lo, hi = _proportion_ci(ind, N_BOOT, rng)
        rows.append({"age_mid_ga": mid, "frac": frac, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).sort_values("age_mid_ga")


# ══════════════════════════════════════════════════════════════════════════════
#  Panel (b)：50 Myr 样本数细柱（左轴）+ 200 Myr 弧比例（右轴）
# ══════════════════════════════════════════════════════════════════════════════

def _draw_panel_b(ax, frames: dict, centers: np.ndarray) -> "plt.Axes":
    """
    同一面板同时表达：50 Myr 采样密度（细柱）与 200 Myr 弧比例（点线）。
      · 左轴 No. of samples（log scale）：灰细柱=全部样品；
        红细柱=P_CA+P_IA+P_IOA(=Arc_probability3) ≥ 0.5 的样品。
      · 右轴 Arc-related affinity (%)：蓝实线空心圆=200 Myr 中 Arc_probability3 ≥ 0.5 的比例；
                                       灰虚线空心三角=Liu et al., 2024 (≥0.5) 外部参照。
    返回右轴 ax2，供主图组装统一处理 x 轴。
    """
    df = frames["df"]
    liu_summary = frames["liu_summary"]

    # 中文注释：右轴两条独立曲线均用 Arc_probability3 ≥ 0.5 口径，并各自计算 95%
    # bootstrap CI——作为"本文数据信号"以实心曲线 + 置信区间呈现，与文献阴影区分。
    geo_ci = _geodan_arc_fraction_ci(df, ARC_THRESHOLD)
    liu_ci = _liu_arc_fraction_ci(liu_summary)

    # 中文注释：样本数改用 50 Myr 细分箱，避免 2.6–2.8 Ga 的样品被合并成单个大柱。
    age_ma = pd.to_numeric(df["C_AGE"], errors="coerce")
    if "AGE" in df.columns:
        age_ma = age_ma.fillna(pd.to_numeric(df["AGE"], errors="coerce"))
    bin_width_ma = 50
    bin_edges_ma = np.arange(2400, 4050, bin_width_ma)
    bin_centers_ga = (bin_edges_ma[:-1] + bin_edges_ma[1:]) / 2000.0
    total_counts, _ = np.histogram(age_ma.dropna().to_numpy(dtype=float), bins=bin_edges_ma)

    # 中文注释：弧类样本改用 P_CA+P_IA+P_IOA(=Arc_probability3) ≥ 0.5 阈值口径，
    # 与右轴弧比例线和 Liu 的 ≥0.5 对齐（不再用 argmax top-1 类别）。
    arc_prob = pd.to_numeric(df["Arc_probability3"], errors="coerce")
    arc_age_ma = age_ma[arc_prob >= ARC_THRESHOLD]
    arc_counts, _ = np.histogram(
        arc_age_ma.dropna().to_numpy(dtype=float),
        bins=bin_edges_ma,
    )

    # 背景窗口 + bin center 线（画在左轴 ax 上）
    _draw_windows(ax)
    _draw_bin_center_lines(ax, centers)

    # ── 左轴：50 Myr log-scale 样本数细柱 + KDE ──
    # 中文注释：log 轴柱子需显式给 base（floor），矩形从 floor 画到样本数。
    floor = 1.0
    ax.set_yscale("log")
    total_positive = total_counts > 0
    arc_positive = arc_counts > 0
    ax.bar(
        bin_centers_ga[total_positive],
        total_counts[total_positive] - floor,
        width=0.046, bottom=floor,
        color=COL_BAR_TOTAL, alpha=0.38,
        edgecolor="#7C7C7C", linewidth=0.35, zorder=2,
        label="Total samples (50 Myr)",
    )
    ax.bar(
        bin_centers_ga[arc_positive],
        arc_counts[arc_positive] - floor,
        width=0.034, bottom=floor,
        color=COL_BAR_ARC, alpha=0.58,
        edgecolor="#9A2E22", linewidth=0.35, zorder=3,
        label="P$_{{arc}}$ ≥ 0.5 (50 Myr)",
    )

    n_max = float(np.nanmax(total_counts))
    ax.set_ylim(floor, n_max * 2.2)
    ax.set_ylabel("No. of samples (log)", fontsize=10.5)
    ax.grid(False)

    # ── 右轴：两条独立弧比例曲线（蓝线带细误差棒；Liu 仅虚线参照，无 CI）──
    ax2 = ax.twinx()
    # Liu 外部参照：灰虚线（不画误差棒/CI 带）
    ax2.plot(liu_ci["age_mid_ga"], liu_ci["frac"], color=COL_LIU,
             linestyle=(0, (5, 3)), linewidth=1.2, marker="^", markersize=6.0,
             markerfacecolor="white", markeredgecolor=COL_LIU, markeredgewidth=1.1,
             alpha=0.9, zorder=6, label="Liu et al., 2024 (≥0.5)")
    # GeoDAN 本文数据信号：蓝实线 + 细误差棒（95% bootstrap CI，仅给蓝线）
    geo_x = geo_ci["age_mid_ga"].to_numpy(dtype=float)
    geo_y = geo_ci["frac"].to_numpy(dtype=float)
    geo_err = np.vstack([geo_y - geo_ci["lo"].to_numpy(dtype=float),
                         geo_ci["hi"].to_numpy(dtype=float) - geo_y])
    ax2.errorbar(geo_x, geo_y, yerr=geo_err, fmt="none", ecolor=COL_LINE_ARC,
                 elinewidth=0.9, capsize=2.2, capthick=0.9, alpha=0.85, zorder=6)
    ax2.plot(geo_x, geo_y, color=COL_LINE_ARC,
             linestyle="-", linewidth=2.0, marker="o", markersize=6.2,
             markerfacecolor="white", markeredgecolor=COL_LINE_ARC,
             markeredgewidth=1.5, zorder=7, label="GeoDAN arc fraction (≥0.5)")
    # 中文注释：右轴固定 0–70%——折线趋势更突出；Liu 峰值(~61%)仍在轴内，
    # 左上角图例仍在数据之上。
    ax2.set_ylim(0.0, 0.7)
    ax2.spines["right"].set_position(("axes", 1.0))
    # ax2.set_ylabel("Arc-related affinity (%)", fontsize=10.5, labelpad=10)
    # 中文注释：百分号仅保留在轴标题中，刻度显示 0、20、40、60。
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6])
    # 中文注释：twinx 会叠加第二套坐标轴；隐藏其左、下、上边框，避免黑色默认轴脊压住主图灰色边框。
    for side in ("left", "bottom", "top"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(SPINE_GRAY)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.spines["right"].set_capstyle("butt")
    ax2.set_ylabel("Arc-related affinity (%)", color="#222222", fontsize=10, labelpad=10)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, position: f"{value * 100:.0f}")
    )
    ax2.tick_params(axis="y", labelsize=9)
    ax2.grid(False)

    # ── 图例只保留柱状；采用无边框单行布局，与 Panel (c) 的轻量风格一致。──
    h1, l1 = ax.get_legend_handles_labels()
    legend = ax2.legend(
        h1, l1, loc="upper center", bbox_to_anchor=(0.554, 0.975),
        fontsize=9, 
        frameon=False, ncol=2,
        handlelength=2.0, handletextpad=0.5,
        columnspacing=1.2, labelspacing=0.25, borderaxespad=0.0,
    )
    legend.set_zorder(10)

    # ── 两条折线在右端空白区直接标注（白色描边光晕，不用矩形框）──
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    ax2.text(2.52, 0.25, "GeoDAN arc fraction (≥0.5)", color=COL_LINE_ARC,
             ha="right", va="center", fontsize=9, fontweight="bold",
             path_effects=halo, zorder=9)
    ax2.text(2.52, 0.565, "Liu et al., 2024 (≥0.5)", color="#555555",
             ha="right", va="center", fontsize=9, fontweight="bold",
             path_effects=halo, zorder=9)

    return ax2


# ══════════════════════════════════════════════════════════════════════════════
#  Panels (c)–(e)：三个原始比值独立小图
# ══════════════════════════════════════════════════════════════════════════════

def _add_subduction_signature_arrow(ax, direction: str,
                                     text_position: str,
                                     y_offset: float = 0.0) -> None:
    """在指标小图左侧添加沿增强方向逐渐加深的低饱和箭头。"""
    if direction == "up":
        start_y, end_y = 0.20, 0.82
        gradient = np.linspace(0.08, 0.72, 256).reshape(-1, 1)
    elif direction == "down":
        start_y, end_y = 0.82, 0.20
        gradient = np.linspace(0.72, 0.08, 256).reshape(-1, 1)
    else:
        raise ValueError(f"未知箭头方向: {direction}")

    # 中文注释：y_offset 使用坐标轴比例单位，只微调箭头和文字的竖向位置。
    start_y += y_offset
    end_y += y_offset

    # 中文注释：先创建单个完整箭头路径，再把渐变图层裁剪到该路径内，
    # 使箭头和箭杆连续一体，避免分开绘制产生重叠横带。
    arrow_shape = mpatches.FancyArrowPatch(
        (0.027, start_y), (0.027, end_y),
        transform=ax.transAxes,
        arrowstyle="simple,head_length=8,head_width=9,tail_width=3.2",
        mutation_scale=1.0,
        facecolor="none", edgecolor="none", linewidth=0.0,
    )
    ax.add_patch(arrow_shape)

    rgba = np.zeros((256, 1, 4), dtype=float)
    rgba[:, :, :3] = matplotlib.colors.to_rgb("#A7C7E7")
    rgba[:, :, 3] = gradient
    gradient_image = ax.imshow(
        rgba, extent=(0.014, 0.040, min(start_y, end_y), max(start_y, end_y)),
        transform=ax.transAxes, origin="lower", aspect="auto",
        interpolation="bicubic", zorder=2.4, clip_on=True,
    )
    gradient_image.set_clip_path(arrow_shape)
    if text_position == "top":
        text_y, va = 0.91, "top"
    elif text_position == "bottom":
        text_y, va = 0.11, "bottom"
    else:
        raise ValueError(f"未知文字位置: {text_position}")
    text_y += y_offset
    ax.text(
        0.012, text_y, "Subduction signature",
        transform=ax.transAxes, ha="left", va=va,
        fontsize=8.0, fontfamily="Arial", fontstyle="italic", color="#222222", zorder=5,
    )


def _draw_indicator_panel(ax, ind_stats: dict[str, pd.DataFrame], centers,
                          metric: str, color: str, marker: str,
                          linestyle, label: str,
                          signature_direction: str,
                          signature_text_position: str,
                          show_subset_legend: bool = False,
                          signature_y_offset: float = 0.0) -> None:
    """使用独立线性 y 轴绘制加权均值及非对称 bootstrap 95% CI。"""
    _draw_windows(ax)
    _draw_bin_center_lines(ax, centers)

    tab = ind_stats[metric]
    x = tab["age_mid_ga"].to_numpy(dtype=float)
    y = tab["weighted_mean"].to_numpy(dtype=float)
    y_lo = tab["bootstrap_ci_lo"].to_numpy(dtype=float)
    y_hi = tab["bootstrap_ci_hi"].to_numpy(dtype=float)

    # 中文注释：点值为 Arc_probability3 加权均值，误差棒为非对称 bootstrap 95% CI。
    valid = np.isfinite(y) & np.isfinite(y_lo) & np.isfinite(y_hi)
    yerr = np.vstack([
        np.maximum(0.0, y[valid] - y_lo[valid]),
        np.maximum(0.0, y_hi[valid] - y[valid]),
    ])
    ax.errorbar(
        x[valid], y[valid], yerr=yerr,
        fmt="none",
        ecolor=color, elinewidth=0.9, capsize=2.2, capthick=0.9,
        alpha=0.55, zorder=3,
    )
    ax.plot(
        x, y, color=color, linestyle=linestyle, linewidth=1.65,
        marker=marker, markersize=5.2, markerfacecolor="white",
        markeredgecolor=color, markeredgewidth=1.25, zorder=4,
        label=r"$P_{\mathrm{arc}}$-weighted mean",
    )

    values = np.concatenate([y_lo, y_hi])
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    # 中文注释：c、d 子图固定较小的 y 轴范围，以突出加权均值曲线的变化。
    # 超出坐标范围的误差棒会在边界处截断；e 子图继续使用原来的自动范围。
    if metric == "Ba_La":
        lower, upper = 0.0, 140.0
    elif metric == "Th_Nb":
        lower, upper = 0.0, 1
    else:
        lower = max(0.0, lo - 0.06 * span)
        upper = hi + 0.06 * span
    ax.set_ylim(lower, upper)

    if metric in {"Ba_La", "Th_Nb"}:
        # 中文注释：固定较小范围时，用三角标记指出误差棒仍向坐标轴外延伸，
        # 避免长误差棒被裁切后看起来像是没有绘制。
        boundary_pad = 0.025 * (upper - lower)
        upper_overflow = valid & (y_hi > upper)
        lower_overflow = valid & (y_lo < lower)
        if np.any(upper_overflow):
            ax.scatter(
                x[upper_overflow],
                np.full(np.count_nonzero(upper_overflow), upper - boundary_pad),
                marker="^", s=20, facecolor="white", edgecolor=color,
                linewidth=0.9, zorder=5, clip_on=True,
            )
        if np.any(lower_overflow):
            ax.scatter(
                x[lower_overflow],
                np.full(np.count_nonzero(lower_overflow), lower + boundary_pad),
                marker="v", s=20, facecolor="white", edgecolor=color,
                linewidth=0.9, zorder=5, clip_on=True,
            )

    ax.set_ylabel(label, fontsize=9.2, color="#222222", labelpad=8)
    # 中文注释：零面板间距下去掉最顶部的边界刻度，避免与上一面板底部刻度重叠。
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="upper"))
    visible_ticks = ax.get_yticks()
    y_min, y_max = ax.get_ylim()
    visible_ticks = visible_ticks[
        (visible_ticks >= y_min) & (visible_ticks <= y_max)
    ]
    if len(visible_ticks) > 1:
        ax.set_yticks(visible_ticks[:-1])
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(False)
    if show_subset_legend:
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.56, 0.98),
            fontsize=7.2, frameon=False, ncol=1,
            handlelength=1.8, columnspacing=1.0,
        )
    _add_subduction_signature_arrow(
        ax, signature_direction, signature_text_position, signature_y_offset,
    )


def _draw_geochemical_panels(ax_c, ax_d, ax_e,
                             ind_stats: dict[str, pd.DataFrame],
                             centers: np.ndarray) -> None:
    """按 Ba/La、Th/Nb、Nb/La 顺序绘制三个独立原始比值小图。"""
    _draw_indicator_panel(
        ax_c, ind_stats, centers, "Ba_La", COL_BA_LA, "D",
        "-", "Ba/La", "up", "top", True,
    )
    _draw_indicator_panel(
        ax_d, ind_stats, centers, "Th_Nb", COL_TH_NB, "s",
        "-", "Th/Nb", "up", "bottom", signature_y_offset=-0.04,
    )
    _draw_indicator_panel(
        ax_e, ind_stats, centers, "Nb_La", COL_NB_LA, "o", "-", "Nb/La",
        "down", "top",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  主图组装
# ══════════════════════════════════════════════════════════════════════════════

def build_main_figure(frames: dict, ind_stats: dict[str, pd.DataFrame]) -> None:
    centers = _bin_center_ages(frames["age_summary"])

    fig = plt.figure(figsize=(8.8, 9.8))
    fig.patch.set_facecolor("#FFFFFF")
    # 中文注释：顶部彩色年代条（客观年代标尺）与 Panel (a) 之间留出间距，
    # 与下方解释性标注（本文解读）在视觉上分层，避免把年代标尺与解读混为一谈。
    gs = fig.add_gridspec(
        5, 1,
        # 中文注释：略微压缩 Panel (a) 和三个地球化学小图，Panel (b) 保持信息空间。
        height_ratios=[3.65, 2.70, 1.05, 1.05, 1.05],
        hspace=0.0, left=0.115, right=0.905, top=0.98, bottom=0.055,
    )
    # 中文注释：年代矩形轴与 Panel (a) 零间距相接，避免两者之间出现白缝。
    gs_top = gs[0].subgridspec(2, 1, height_ratios=[0.42, 3.00], hspace=0.0)
    ax_top = fig.add_subplot(gs_top[0])
    ax_a   = fig.add_subplot(gs_top[1])
    # 中文注释：所有数据面板共享同一 x 轴，确保左右边界和时间刻度完全重合。
    ax_b   = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c   = fig.add_subplot(gs[2], sharex=ax_a)
    ax_d   = fig.add_subplot(gs[3], sharex=ax_a)
    ax_e   = fig.add_subplot(gs[4], sharex=ax_a)

    _draw_top_strip(ax_top)

    _draw_panel_a(ax_a, frames["df"], frames["class_names"], centers)
    _style_panel(ax_a, show_xlabel=False)

    _draw_panel_b(ax_b, frames, centers)          # 内部创建右轴 twin
    _style_panel(ax_b, show_xlabel=False)
    ax_b.tick_params(axis="x", bottom=False, labelbottom=False)

    _draw_geochemical_panels(ax_c, ax_d, ax_e, ind_stats, centers)
    _style_panel(ax_c, show_xlabel=False)
    _style_panel(ax_d, show_xlabel=False)
    _style_panel(ax_e, show_xlabel=True)

    # 中文注释：y 轴标题和 a-e 子图编号使用两个独立横坐标，避免误改导致版面串动。
    y_label_x = -0.065
    panel_label_x = -0.072

    # 中文注释：固定五个左侧 y 轴标题的横向坐标，避免不同刻度宽度造成错位。
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        ax.yaxis.set_label_coords(y_label_x, 0.5)

    # 中文注释：零面板间距下，编号统一向右微调，避免压到上一面板。
    for ax, lab, y_pos in [
        (ax_a, "a", 1.0), (ax_b, "b", 0.985),
        (ax_c, "c", 0.985), (ax_d, "d", 0.988), (ax_e, "e", 0.985),
    ]:
        ax.text(panel_label_x, y_pos, lab, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", ha="left", zorder=12)

    FIG_MAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_MAIN_PATH, dpi=1200, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[OK] 主图已保存: {FIG_MAIN_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  Panel (c) 附图：加权均值 + log-scale y 轴
# ══════════════════════════════════════════════════════════════════════════════

def build_panel_c_raw(frames: dict, ind_stats: dict[str, pd.DataFrame]) -> None:
    centers = _bin_center_ages(frames["age_summary"])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    fig.patch.set_facecolor("#FFFFFF")

    _draw_windows(ax)
    _draw_bin_center_lines(ax, centers)

    series = [
        ("Ba_La", COL_BA_LA, "D", (0, (1, 1.4)), "Ba/La"),
        ("Th_Nb", COL_TH_NB, "s", (0, (5, 2)),   "Th/Nb"),
        ("Nb_La", COL_NB_LA, "o", "-",           "Nb/La"),
    ]
    for metric, color, marker, ls, label in series:
        tab = ind_stats[metric]
        x = tab["age_mid_ga"].to_numpy(dtype=float)
        y = tab["weighted_mean"].to_numpy(dtype=float)
        y_lo = tab["bootstrap_ci_lo"].to_numpy(dtype=float)
        y_hi = tab["bootstrap_ci_hi"].to_numpy(dtype=float)
        yerr = np.vstack([
            np.maximum(0.0, y - y_lo),
            np.maximum(0.0, y_hi - y),
        ])
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.8,
                    capsize=2.0, capthick=0.8, alpha=0.85, zorder=3)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.6, marker=marker,
                markersize=5.6, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.3, zorder=4, label=label)

    ax.set_yscale("log")
    ax.set_xlim(X_OLD, X_YOUNG)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Age (Ga)", fontsize=10.5)
    ax.set_ylabel("Weighted mean elemental ratio (log scale)", fontsize=10.5)
    _apply_boxed_panel(ax)
    # 中文注释：附加导出的 Panel (c) 同样不显示格网。
    ax.grid(False)
    # 中文注释：附图沿用主图 Panel (c) 的轻量无边框图例风格。
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.50, 0.985),
        fontsize=8.0, frameon=False, ncol=3,
        handlelength=2.2, handletextpad=0.45,
        columnspacing=1.25, borderaxespad=0.0,
    )
    # 中文注释：log 轴下 Th/Nb 落在底部，误差说明放到 Ba/La 与 Th/Nb 之间的空白带，
    # 并省略底部 transition 细条，避免与底部曲线重叠（该背景已由窗口阴影表达）。
    ax.text(0.992, 0.40, r"Error bars: $\pm 2$ bootstrap SEM", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.0, color="#555555")

    fig.subplots_adjust(left=0.10, right=0.975, top=0.95, bottom=0.135)
    fig.savefig(FIG_PANELC_RAW, dpi=600, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[OK] Panel (c) 附图已保存: {FIG_PANELC_RAW}")


# ══════════════════════════════════════════════════════════════════════════════
#  建议图注（写入 sidecar 文本，供正文/图注使用——不画进图里）
# ══════════════════════════════════════════════════════════════════════════════

CAPTION_TEXT = """\
Archean time-evolution synthesis of tectonic affinity and geochemistry
(4.0 → 2.5 Ga, old to young), in which externally literature-constrained tectonic
context and this study's data signals are deliberately drawn in two different
visual languages. The thin pale-orange line beneath the geological-period strip
marks the ~3.2–2.5 Ga Meso–Neoarchean transition context constrained by external
literature (Cawood et al., 2018); it is contextual rather than a result of this
study. The green ~2.7–2.5 Ga band marks the arc-affinity rise observed in panel
(b), consistent with the late-Archean cratonization interval reported in previous
literature. Its overlap with the tail of the broader transition context represents
a specific data-observed enhancement nested within the known geological framework.
The narrow ~3.8 Ga shaded strip marks a weak arc-like signal seen only in a few
terranes such as Greenland and Pilbara (Polat & Hofmann, 2003; Furnes et al.,
2009). The ~3.5 Ga "transient arc pulse" is shown as a point annotation and is
corroborated by two independent curves in panel (b). Faint vertical dashed lines
mark 200 Myr bin centres; 200 Myr bins accommodate
Archean age uncertainty and sample sparsity and allow comparison with the binned
proxies of Liu et al. (2024).
(a) Filled KDE ridgelines of the nine tectonic environments (CA, IA, IOA, BAB, OP,
OI, CF, MOR, CR); age-bin proportions are interpolated and smoothed under one global
height scale (no within-class peak normalization), so ridge heights retain
between-class differences and the overall temporal trend.
(b) Sample density on a log left axis and arc fraction on the right axis. Grey/red
bars are all samples / samples with Arc_probability3 (= P_CA+P_IA+P_IOA) ≥ 0.5 in
50 Myr bins. The solid blue curve is the GeoDAN arc fraction (≥0.5) and the grey
dashed curve is the Liu et al. (2024) binary-arc reference (≥0.5), both in 200 Myr
bins, each shown with a 95% bootstrap confidence band; their synchronous local
maximum at ~3.5 Ga supports the transient arc pulse.
(c–e) Arc_probability3-weighted bootstrap means of Ba/La, Th/Nb and Nb/La,
respectively, calculated in the same 200 Myr bins. Samples with stronger modelled
arc affinity receive greater weight. All valid GeoDAN samples are retained without
a general Ba concentration cutoff, except for the anomalous sample s_2C-14 [24900]
with Ba approximately 23349 ppm. Error bars show the non-symmetric 2.5th–97.5th percentile
interval of the bootstrap weighted-mean distribution. These curves are a
model-conditioned consistency diagnostic rather than independent validation of the classifier.
Each proxy uses its own linear y-axis. Pale-blue arrows indicate the direction
of increasing subduction signature for each proxy.
Visual-language key: thin pale-orange line = externally constrained transition
context; green band and plotted curves = this study's data-observed signal; the
~3.5 Ga point annotation and ~3.8 Ga narrow band mark additional arc-affinity
features supported by the data.
Note: the Liu et al. (2024) ≥0.5 threshold is a binary external reference for trend
comparison only, not an absolute proportion; GeoDAN's nine-class framework separates
transitional arc-like signals (e.g. BAB) from typical CA/IA/IOA arc end-members, so
a lower absolute arc fraction than Liu does not indicate a contradiction but a more
conservative end-member partition.
"""


def write_caption() -> None:
    CAPTION_TXT_PATH.write_text(CAPTION_TEXT, encoding="utf-8")
    print(f"[OK] 建议图注已保存: {CAPTION_TXT_PATH}")


def write_indicator_audit(ind_stats: dict[str, pd.DataFrame]) -> None:
    """导出加权均值、bootstrap 95% CI、SEM与有效样本量，便于复核图中每一个点。"""
    rows = []
    for metric, table in ind_stats.items():
        metric_table = table.copy()
        metric_table.insert(0, "metric", metric)
        rows.append(metric_table)
    audit = pd.concat(rows, ignore_index=True)
    INDICATOR_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(INDICATOR_AUDIT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 地球化学指标审计表已保存: {INDICATOR_AUDIT_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 78)
    print("太古代时间演化图（共享时间轴多证据对齐图）")
    print("=" * 78)
    frames = load_core_frames()
    ind_stats = compute_indicator_stats(frames["df"])
    build_main_figure(frames, ind_stats)
    build_panel_c_raw(frames, ind_stats)
    write_caption()
    write_indicator_audit(ind_stats)
    print("完成。既有输出文件未改动，可随时回退对比。")


if __name__ == "__main__":
    main()
