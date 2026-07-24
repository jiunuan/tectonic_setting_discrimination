"""
CFB=6920、缺失编码GeoDAN最终方案的现代训练集与太古代应用集PCA分析。

数据流：
- 现代玄武岩：读取训练集插补后、主量无水标准化后的连续数据。
- 太古代玄武岩：从3483条年龄非空候选样品按无水SiO2 44-53 wt%
  和MgO不大于18 wt%筛选为3012条，不执行地球化学插补。
- PCA 不使用 1-255 分位数分箱值，保留真实地球化学含量关系。
- PCA 轴只用现代训练集拟合；太古代样品只投影到固定 PCA 空间。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_MAJOR_NORM_CSV,
    ARCHEAN_POOL_CSV,
    ARCHEAN_FINAL_PREDICTIONS_CSV,
    ZENODO_ARCHEAN_CSV,
    ZENODO_ARCHEAN_PREDICTIONS_CSV,
    ARCHEAN_CONSISTENCY_DIR,
)

from archean_s3_preprocess import preprocess_archean

# 中文注释：当前项目通过集中配置选择正式数据或等价的 Zenodo 发布表。
RAW_MODERN_PATH = Path(TRAIN_MAJOR_NORM_CSV)
ARCHEAN_RAW_PATH = Path(ZENODO_ARCHEAN_CSV if Path(ZENODO_ARCHEAN_CSV).exists() else ARCHEAN_POOL_CSV)
FINAL_PREDICTION_PATH = Path(ZENODO_ARCHEAN_PREDICTIONS_CSV if Path(ZENODO_ARCHEAN_PREDICTIONS_CSV).exists() else ARCHEAN_FINAL_PREDICTIONS_CSV)
OUT_ROOT = Path(ARCHEAN_CONSISTENCY_DIR)

CFB_TARGET_COUNT = 6920

# =========================
# 元素 / 标签 / 常量
# =========================
MAJORS = ["SIO2", "TIO2", "AL2O3", "FEOT", "MNO", "MGO", "CAO", "NA2O", "K2O", "P2O5"]
TRACES = ["RB", "V", "CR", "CO", "NI", "BA", "SR", "Y", "ZR", "NB",
          "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO",
          "ER", "YB", "LU", "HF", "TA", "TH"]
ALL_ELEMENTS = MAJORS + TRACES

TECTONIC_COL = "TECTONIC SETTING"
HIGH_CONF_THRESHOLD = 0.50   # 中文注释：高弧亲和性阈值 P_arc >= 0.50。
LOW_CONF_THRESHOLD = 0.30    # 中文注释：低弧亲和性阈值 P_arc < 0.30。
GEODAN_ARC_LABELS = ["Continental arc", "Island arc", "Intra-oceanic arc"]
RANDOM_SEED = 42
CHI2_95_DF2 = 5.991464547107979
CHI2_99_DF2 = 9.210340371976294
ELLIPSE_95_SCALE = np.sqrt(CHI2_95_DF2)
ELLIPSE_99_SCALE = np.sqrt(CHI2_99_DF2)

# 中文注释：applicability-domain kNN 距离的参数。
AD_VAR_TARGET = 0.85   # 累计解释方差阈值，决定保留多少个 PC（约 12 个 -> 85% 方差）。
AD_K = 10              # kNN 的 k 值。

# 中文注释：(a) 子图 envelope 椭圆的视觉收紧系数（<1 让圆环更小、更贴主云团）。
ENV_VIS_SCALE = 0.75

# =========================
# applicability-domain 三联图配色（低饱和，期刊主文风格）
# =========================
# (a) PCA reference space
C_MODERN_FILL = "#A7B2BA"   # 现代参考点云中浅灰（够深才看得清）
C_ARCHEAN = "#4A6C88"       # 太古代钢蓝（比原灰蓝更明显）
C_HIGH_ARC = "#D84A3A"      # 高弧亲和性使用低饱和朱红，突出但不刺眼
C_ENV_95 = "#36586A"        # 95% envelope 深岩蓝
C_ENV_99 = "#5C7E90"        # 99% envelope 稍浅岩蓝
# (b) applicability-domain distance
C_MODERN_OUTLINE = "#4C555C"  # 现代参考距离 step 轮廓
# 中文注释：三段按「距现代域越来越远」做感知有序渐变 蓝(域内)→琥珀(边缘)→红(越界)。
C_IN_DOMAIN = "#3F7CA3"     # ≤95% 域内 蓝
C_MARGINAL = "#E2A33C"      # 95–99% 边缘 琥珀
C_OUT_DOMAIN = "#B45A50"    # >99% 越界使用更暗的砖红/棕红
C_THRESH = "#30343A"        # 阈值竖线
# (c) quantitative coverage
C_CONNECT = "#B8C0C7"       # dumbbell 连线
C_COV95 = "#31566B"         # ≤95% 覆盖率深蓝灰圆点
C_COV99 = "#A9BFCC"         # ≤99% 覆盖率浅蓝灰方块
C_COV99_EDGE = "#5F7888"    # ≤99% 方块描边
C_NLABEL = "#5F666D"        # 样品数标签深灰色
# 通用
C_TEXT = "#000000"          # 图内注释主色：黑色
C_SPINE = "#000000"         # 坐标轴边框：黑色
C_GRID = "#E9ECEF"          # 极浅网格线

CLASS_ABBREVS = {
    "CONTINENTAL ARC": "CA", "INTRA-OCEANIC ARC": "IOA", "ISLAND ARC": "IA",
    "BACK-ARC_BASIN": "BAB", "SPREADING_CENTER": "MOR", "OCEANIC PLATEAU": "OP",
    "OCEAN ISLAND": "OI", "CONTINENTAL FLOOD BASALT": "CFB", "CONTINENTAL_RIFT": "CR",
}

# 中文注释：整体标注、刻度、轴标签和图例字号较原版上调 2 pt。
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.linewidth": 0.9, "axes.labelsize": 14, "axes.titlesize": 14,
    "axes.edgecolor": "#000000",
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "xtick.color": "#000000", "ytick.color": "#000000",
    "axes.labelcolor": "#000000", "text.color": "#000000",
    "figure.dpi": 100, "savefig.dpi": 600, "savefig.bbox": "tight",
})


# =========================
# 通用工具
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名里的 (WT%) / (PPM) 后缀。"""
    return df.rename(columns={c: c.replace("(WT%)", "").replace("(PPM)", "").strip() for c in df.columns})


def abbr(cls: str) -> str:
    return CLASS_ABBREVS.get(cls, cls)


def extract_elements(df: pd.DataFrame) -> pd.DataFrame:
    """从 df 抽出 36 个元素列，缺失列填 NaN。"""
    out = pd.DataFrame(index=df.index)
    for col in ALL_ELEMENTS:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    return out[ALL_ELEMENTS]


# =========================
# PCA 输入构造
# =========================
def to_pca_space(train: pd.DataFrame, arch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """把无水标准化后的现代/太古代特征转到 log10 + 现代训练集标准化空间，不做插值。"""
    # 中文注释：0 和负值视为缺失；用现代训练集 nanmean/nanstd 标准化。
    # 中文注释：缺失值填到标准化空间的 0，也就是现代训练集均值位置。
    tr, ar = train.copy(), arch.copy()
    tr[tr <= 0], ar[ar <= 0] = np.nan, np.nan
    tlog, alog = np.log10(tr.to_numpy(dtype=float)), np.log10(ar.to_numpy(dtype=float))
    mean = np.nanmean(tlog, axis=0)
    std = np.nanstd(tlog, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (np.nan_to_num((tlog - mean) / std, nan=0.0),
            np.nan_to_num((alog - mean) / std, nan=0.0))


# =========================
# applicability-domain：多维 PCA 子空间里的 kNN 距离
# =========================
def compute_ad_distances(PC_train_full: np.ndarray, PC_arch_full: np.ndarray,
                         var_ratio: np.ndarray, var_target: float = AD_VAR_TARGET,
                         k: int = AD_K) -> dict:
    """在累计解释 >= var_target 方差的前若干个 PC 子空间里，计算每个样品到现代训练集的 kNN 平均距离。

    - 子空间维度 n_pc 由累计解释方差决定（约 12 个 PC -> 85% 方差）。
    - 各 PC 用现代训练集的标准差白化，使每个保留维度等权（等价于子空间内的马氏距离）。
    - 现代训练集用 leave-one-out（排除自身）得到自身 kNN 距离分布，据此定义 95% / 99% 阈值。
    """
    cum = np.cumsum(var_ratio)
    n_pc = int(np.searchsorted(cum, var_target) + 1)
    n_pc = max(2, min(n_pc, PC_train_full.shape[1]))

    Wt = PC_train_full[:, :n_pc]
    sd = Wt.std(axis=0)
    sd[sd == 0] = 1.0
    Wt = Wt / sd
    Wa = PC_arch_full[:, :n_pc] / sd

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Wt)
    d_tr, _ = nn.kneighbors(Wt)
    dist_modern = d_tr[:, 1:].mean(axis=1)              # 中文注释：丢掉第 0 列（自身距离 0）。
    d_ar, _ = nn.kneighbors(Wa, n_neighbors=k)
    dist_arch = d_ar.mean(axis=1)

    q95 = float(np.quantile(dist_modern, 0.95))
    q99 = float(np.quantile(dist_modern, 0.99))
    return {
        "n_pc": n_pc, "cum_var": float(cum[n_pc - 1]), "k": k,
        "dist_modern": dist_modern, "dist_arch": dist_arch,
        "q95": q95, "q99": q99,
    }


def coverage_rows(dist_arch: np.ndarray, q95: float, q99: float,
                  high_mask: np.ndarray | None, low_mask: np.ndarray | None) -> list[dict]:
    """三组太古代样品（全部 / 高弧亲和 / 低弧亲和）在现代 95% / 99% 域内的覆盖率。"""
    groups = [("All", np.ones(len(dist_arch), dtype=bool))]
    if high_mask is not None:
        groups.append(("High arc", high_mask))
    if low_mask is not None:
        groups.append(("Low arc", low_mask))
    rows = []
    for label, m in groups:
        d = dist_arch[m]
        rows.append({
            "group": label, "n": int(m.sum()),
            "cov95": round(100 * float(np.mean(d <= q95)), 1),
            "cov99": round(100 * float(np.mean(d <= q99)), 1),
        })
    return rows


# =========================
# 绘图
# =========================
def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """二维点云的置信椭圆。"""
    if len(x) < 5:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2,
                      height=np.sqrt(1 - pearson) * 2, **kwargs)
    transf = (mpl.transforms.Affine2D().rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _panel_tag(ax, tag: str, x_pts: float = -52, y_pts: float = -12) -> None:
    """子图左上角外侧加粗标注 (a)/(b)/(c)。

    用「相对轴左上角的固定点数偏移」定位，与面板宽高无关——这样不同高度/宽度的子图
    标号离各自顶边、左边的绝对距离一致（顶边对齐的 (a)/(b) 标号高度相同）。
    中文注释：子图标题移除后，将编号收近到绘图区，减少顶部和左侧的无效留白。
    x_pts 越负越靠左，y_pts 越大越靠上。
    """
    ax.annotate(tag, xy=(0, 1), xycoords="axes fraction",
                xytext=(x_pts, y_pts), textcoords="offset points",
                ha="left", va="bottom", fontsize=20, fontweight="bold", color=C_TEXT)


def _style_axes(ax) -> None:
    """统一显示四条边界线，边框和刻度使用同一套浅深灰。"""
    # 中文注释：PCA 主图的每个子图都保留上、下、左、右四条边界线，避免不同面板边框风格不一致。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_SPINE)
        spine.set_linewidth(0.5)
        spine.set_capstyle("butt")
    ax.tick_params(length=3, width=0.7, color=C_SPINE, labelcolor="#000000")


def _panel_a_map(ax, PC_train, PC_arch, high_mask, ad, var1, var2):
    """(a) Modern basalt reference space：现代参考点云 + 95/99 置信椭圆 envelope + 太古代叠加。"""
    xt, yt = PC_train[:, 0], PC_train[:, 1]

    # 中文注释：所有现代玄武岩合并成中浅灰参考云，点径小、透明度适中，看得清又不压住太古代点。
    ax.scatter(xt, yt, s=6, c=C_MODERN_FILL, alpha=0.20, edgecolors="none", zorder=1)

    # 中文注释：现代训练域 95%（实线）/ 99%（长虚线）置信椭圆 envelope，低饱和细线、点在其上。
    confidence_ellipse(xt, yt, ax, n_std=ELLIPSE_99_SCALE, edgecolor=C_ENV_99,
                       facecolor="none", lw=1.0, ls=(0, (5, 4)), alpha=0.75, zorder=3)
    confidence_ellipse(xt, yt, ax, n_std=ELLIPSE_95_SCALE, edgecolor=C_ENV_95,
                       facecolor="none", lw=1.2, alpha=0.85, zorder=3)

    # 中文注释：太古代样品；每颗点加极细白描边，让重叠点彼此分离、产生层次/立体感。
    #          高弧亲和性单独正红高亮、点更大白边更粗，浮在最上层。
    if high_mask is not None:
        ax.scatter(PC_arch[~high_mask, 0], PC_arch[~high_mask, 1], s=17, c=C_ARCHEAN,
                   alpha=0.80, edgecolors="white", linewidths=0.35, zorder=4, label="Archean")
        ax.scatter(PC_arch[high_mask, 0], PC_arch[high_mask, 1], s=30, c=C_HIGH_ARC,
                   alpha=0.92, edgecolors="white", linewidths=0.55, zorder=6,
                   label="High arc-affinity")
    else:
        ax.scatter(PC_arch[:, 0], PC_arch[:, 1], s=17, c=C_ARCHEAN,
                   alpha=0.80, edgecolors="white", linewidths=0.35, zorder=4, label="Archean")

    # ax.text(0.025, 0.97, "Modern basalt reference space", transform=ax.transAxes,
    #         ha="left", va="top", fontsize=11.5, style="italic", color=C_TEXT)
    ax.set_xlabel(f"PC1 (modern-reference PCA; {var1 * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2 * 100:.1f}% variance)")

    # 中文注释：坐标范围放宽到现代点 ~4.4σ，让 95/99 envelope 留出余白、看起来更舒展、
    #          不顶到图框；同时与太古代点 99% 范围取并集，保证 99% 椭圆和太古代点都不被裁。
    mx, my = xt.mean(), yt.mean()
    sx, sy = xt.std(), yt.std()
    ax_x = np.percentile(np.abs(PC_arch[:, 0] - mx), 99) + 0.5 * sx
    ax_y = np.percentile(np.abs(PC_arch[:, 1] - my), 99) + 0.5 * sy
    rx = max(4.4 * sx, ELLIPSE_99_SCALE * sx + 0.8 * sx, ax_x)
    ry = max(4.4 * sy, ELLIPSE_99_SCALE * sy + 0.8 * sy, ax_y)
    ax.set_xlim(mx - rx, mx + rx)
    ax.set_ylim(my - ry, my + ry)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MODERN_FILL,
               markeredgecolor="#8C969D", markersize=6, label="Modern reference"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ARCHEAN,
               markeredgecolor="none", markersize=6, label="Archean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HIGH_ARC,
               markeredgecolor="white", markersize=6.5, label="High arc-affinity"),
        Line2D([0], [0], color=C_ENV_95, lw=1.1, label="95% envelope"),
        Line2D([0], [0], color=C_ENV_99, lw=1.0, ls=(0, (5, 4)), label="99% envelope"),
    ]
    if high_mask is None:
        handles = [h for h in handles if h.get_label() != "High arc-affinity"]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="#D0D4D8", fontsize=11, handletextpad=0.5,
                    borderpad=0.6, labelspacing=0.45)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(20)  # 中文注释：图例置于所有散点之上，避免被高 zorder 的橙色点遮挡。
    _panel_tag(ax, "a")


def _panel_b_distance(ax, ad):
    """(b) Applicability-domain distance：太古代到现代域的 kNN 距离直方图 + 95/99 阈值。"""
    dist_arch, dist_modern = ad["dist_arch"], ad["dist_modern"]
    q95, q99 = ad["q95"], ad["q99"]

    xmax = max(q99 * 1.8, np.quantile(dist_arch, 0.97))
    bins = np.linspace(0, xmax, 46)

    # 中文注释：太古代直方图，按 ≤95% / 95–99% / >99% 三段着色。
    counts, edges, patches = ax.hist(dist_arch, bins=bins, density=True, zorder=3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for c, p in zip(centers, patches):
        p.set_facecolor(C_IN_DOMAIN if c <= q95 else C_MARGINAL if c <= q99 else C_OUT_DOMAIN)
        p.set_edgecolor("white")
        p.set_linewidth(0.3)
        p.set_alpha(0.85)

    # 中文注释：现代训练集自身距离分布用细深灰阶梯轮廓叠加作参照。
    ax.hist(dist_modern, bins=bins, density=True, histtype="step",
            color=C_MODERN_OUTLINE, lw=1.0, zorder=4)

    ymax = ax.get_ylim()[1]
    ax.axvline(q95, color=C_THRESH, lw=1.05, ls="-", zorder=5)
    ax.axvline(q99, color=C_THRESH, lw=1.05, ls=(0, (4, 3)), zorder=5)
    ax.text(q95, ymax * 0.97, "95%", rotation=90, va="top", ha="right",
            fontsize=12, color=C_THRESH)
    ax.text(q99, ymax * 0.97, "99%", rotation=90, va="top", ha="right",
            fontsize=12, color=C_THRESH)

    handles = [
        Line2D([0], [0], color=C_MODERN_OUTLINE, lw=1.0, label="Modern reference"),
        mpl.patches.Patch(facecolor=C_IN_DOMAIN, edgecolor="none", label="Archean ≤95%"),
        mpl.patches.Patch(facecolor=C_MARGINAL, edgecolor="none", label="95–99%"),
        mpl.patches.Patch(facecolor=C_OUT_DOMAIN, edgecolor="none", label=">99%"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.85,
                    edgecolor="#D8DCDF", fontsize=10, handlelength=1.1,
                    handletextpad=0.5, labelspacing=0.3, borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("kNN distance to modern reference")
    ax.set_ylabel("Density")
    # ax.set_title("Applicability-domain distance", fontsize=11, pad=6, color=C_TEXT)
    ax.grid(True, axis="y", color=C_GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    _panel_tag(ax, "b")


def _panel_c_coverage(ax, cov_rows):
    """(c) Quantitative coverage：三组太古代在现代 95%/99% 域内的覆盖率 dumbbell plot。"""
    n = len(cov_rows)
    ys = np.arange(n)[::-1]  # 中文注释：第一组放最上面。
    for y, r in zip(ys, cov_rows):
        ax.plot([r["cov95"], r["cov99"]], [y, y], color=C_CONNECT, lw=1.3, zorder=2)
        ax.scatter(r["cov95"], y, s=52, marker="o", color=C_COV95, zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.scatter(r["cov99"], y, s=64, marker="s", color=C_COV99, zorder=3,
                   edgecolors=C_COV99_EDGE, linewidths=0.6)
        lo, hi = sorted([r["cov95"], r["cov99"]])
        ax.text(lo - 1.3, y, f"{lo:.1f}", va="center", ha="right", fontsize=12, color=C_TEXT)
        ax.text(hi + 1.3, y, f"{hi:.1f}", va="center", ha="left", fontsize=12, color=C_TEXT)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["group"] for r in cov_rows], fontsize=12)
    # 中文注释：样品数放在类别 ytick 标签下方，使用更小字号和浅灰色弱化层级。
    for y, r in zip(ys, cov_rows):
        ax.text(-0.035, y - 0.22, f"n={r['n']:,}", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=10, color=C_NLABEL)

    all_vals = [v for r in cov_rows for v in (r["cov95"], r["cov99"])]
    # 中文注释：覆盖率点及其左右数值标签都必须位于坐标范围内，动态预留边距。
    x_margin = 6.0
    x_min = max(0.0, np.floor((min(all_vals) - x_margin) / 5.0) * 5.0)
    x_max = min(105.0, np.ceil((max(all_vals) + x_margin) / 5.0) * 5.0)
    ax.set_xlim(x_min, x_max)
    # 中文注释：减小 y 轴上边界，让三条 coverage 线和 ytick 标签整体视觉上移。
    ax.set_ylim(-0.60, n - 0.25)
    ax.set_xlabel("Coverage within modern domain (%)")
    # ax.set_title("Quantitative coverage", fontsize=11, pad=6, color=C_TEXT)
    ax.grid(True, axis="x", color=C_GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_COV95,
               markeredgecolor="white", markersize=7.5, label="≤95%"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_COV99,
               markeredgecolor=C_COV99_EDGE, markersize=7.5, label="≤99%"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                    ncol=2, frameon=True, framealpha=0.9, edgecolor="#D8DCDF",
                    fontsize=12, handletextpad=0.4, columnspacing=1.4, borderpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    _panel_tag(ax, "c")

def plot_applicability_domain(PC_train, PC_arch, high_mask, ad, cov_rows,
                              var1, var2, paths):
    """Applicability-domain / PCA coverage 三联图：左 (a) 大，右上 (b)、右下 (c)。"""
    fig = plt.figure(figsize=(13.2, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.1], height_ratios=[0.92, 1.08],
                          # 中文注释：右下 c 图略高一点，缓解底部文字和坐标轴的拥挤。
                          # 中文注释：四边框后右列标签更显眼，适当拉开 a 与 b/c 的横向间距。
                          wspace=0.22, hspace=0.38)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    _panel_a_map(ax_a, PC_train, PC_arch, high_mask, ad, var1, var2)
    _panel_b_distance(ax_b, ad)
    _panel_c_coverage(ax_c, cov_rows)

    for p in paths:
        fig.savefig(p, dpi=600, bbox_inches="tight",
                    facecolor="white", transparent=False)
        print(f"    Saved: {p}")
    plt.close(fig)


def plot_per_class(PC_train, PC_arch, labels, classes, colors, high_mask,
                   var1, var2, variant, path):
    n = len(classes)
    ncols, nrows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    pad = 0.8
    xlim = (PC_train[:, 0].min() - pad, PC_train[:, 0].max() + pad)
    ylim = (PC_train[:, 1].min() - pad, PC_train[:, 1].max() + pad)

    for i, cls in enumerate(classes):
        ax = axes[i]
        m = labels == cls

        ax.scatter(PC_train[~m, 0], PC_train[~m, 1], c="lightgrey",
                   s=3, alpha=0.12, edgecolors="none")
        ax.scatter(PC_train[m, 0], PC_train[m, 1], c=[colors[cls]],
                   s=8, alpha=0.45, edgecolors="none")

        if m.sum() >= 20:
            confidence_ellipse(PC_train[m, 0], PC_train[m, 1], ax, n_std=ELLIPSE_95_SCALE,
                               edgecolor="black", facecolor="none", lw=1.3, ls="--", alpha=0.8)

        if high_mask is not None:
            ax.scatter(PC_arch[~high_mask, 0], PC_arch[~high_mask, 1], c="lightgrey",
                       s=9, alpha=0.4, edgecolors="black", linewidths=0.2, zorder=3)
            ax.scatter(PC_arch[high_mask, 0], PC_arch[high_mask, 1], c="crimson",
                       s=14, alpha=0.75, edgecolors="black", linewidths=0.3, zorder=4)
        else:
            ax.scatter(PC_arch[:, 0], PC_arch[:, 1], c="crimson",
                       s=14, alpha=0.7, edgecolors="black", linewidths=0.3, zorder=4)

        ax.set_title(f"{abbr(cls)}\nmodern n = {m.sum():,}", fontsize=12)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.axhline(0, color="grey", lw=0.4, ls="--", alpha=0.4)
        ax.axvline(0, color="grey", lw=0.4, ls="--", alpha=0.4)
        ax.grid(True, alpha=0.15)
        if i % ncols == 0:
            ax.set_ylabel(f"PC2 ({var2 * 100:.1f}%)")
        if i // ncols == nrows - 1:
            ax.set_xlabel(f"PC1 ({var1 * 100:.1f}%)")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey", markersize=6, label="Other modern classes"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=6, label="Current class"),
        Line2D([0], [0], color="black", ls="--", lw=1.3, label="95% confidence ellipse"),
    ]
    if high_mask is not None:
        legend += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
                   markeredgecolor="black", markersize=7, label="Archean low confidence"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                   markeredgecolor="black", markersize=7, label=f"Archean arc-prob >= {HIGH_CONF_THRESHOLD}"),
        ]
    else:
        legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                             markeredgecolor="black", markersize=7, label="Archean samples"))

    fig.legend(handles=legend, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)
    fig.suptitle(f"Per-class PCA panels - {variant}", fontsize=13, y=1.00)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {path}")


# =========================
# 椭圆内占比统计
# =========================
def compute_overlap(PC_train, PC_arch, labels, classes,
                    high_mask, low_mask) -> pd.DataFrame:
    rows = []
    for cls in classes:
        m = labels == cls
        if m.sum() < 20:
            continue
        mu = PC_train[m].mean(axis=0)
        cov = np.cov(PC_train[m].T)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        diff = PC_arch - mu
        inside = np.sum(diff @ inv * diff, axis=1) < CHI2_95_DF2

        row = {
            "class_full": cls, "class_abbr": abbr(cls), "modern_n": int(m.sum()),
            "archean_inside_all_n": int(inside.sum()),
            "archean_inside_all_pct": round(100 * inside.sum() / len(PC_arch), 2),
        }
        if high_mask is not None:
            row["archean_inside_high_n"] = int((inside & high_mask).sum())
            row["archean_inside_high_pct"] = round(100 * (inside & high_mask).sum() / high_mask.sum(), 2)
        if low_mask is not None:
            # 中文注释：低弧组严格使用 P_arc < 0.30，不把中间概率样品混入低弧统计。
            row["archean_inside_low_n"] = int((inside & low_mask).sum())
            row["archean_inside_low_pct"] = round(100 * (inside & low_mask).sum() / low_mask.sum(), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("modern_n", ascending=False)


# =========================
# 无插值版本主流程
# =========================
def run_continuous_pca(train_feats: pd.DataFrame, arch_feats: pd.DataFrame,
                       labels: np.ndarray, classes: list[str], colors: dict,
                       high_mask: np.ndarray | None, low_mask: np.ndarray | None,
                       arch_conf: np.ndarray | None) -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"\n{'=' * 80}\n"
        "Variant: cfb6920_missing_mask\n"
        f"Output : {OUT_ROOT}\n"
        "Mode   : continuous geochemical values"
    )

    # 中文注释：拟合完整 PCA；前 2 个 PC 用于 (a) 散点，前若干个 PC（~85% 方差）用于 kNN 距离。
    X_train, X_arch = to_pca_space(train_feats, arch_feats)
    pca = PCA(random_state=RANDOM_SEED).fit(X_train)
    PC_train_full = pca.transform(X_train)
    PC_arch_full = pca.transform(X_arch)
    PC_train, PC_arch = PC_train_full[:, :2], PC_arch_full[:, :2]
    var1, var2 = pca.explained_variance_ratio_[:2]
    print(f"  PC1: {var1 * 100:.1f}%  PC2: {var2 * 100:.1f}%  PC1+PC2: {(var1 + var2) * 100:.1f}%")

    # 中文注释：在多维 PCA 子空间里计算 applicability-domain kNN 距离与覆盖率。
    print("  Computing applicability-domain kNN distances ...")
    ad = compute_ad_distances(PC_train_full, PC_arch_full, pca.explained_variance_ratio_)
    cov_rows = coverage_rows(ad["dist_arch"], ad["q95"], ad["q99"], high_mask, low_mask)
    print(f"  AD subspace: {ad['n_pc']} PCs ({ad['cum_var'] * 100:.1f}% var), k={ad['k']}, "
          f"q95={ad['q95']:.3f}, q99={ad['q99']:.3f}")
    for r in cov_rows:
        print(f"    {r['group'].splitlines()[0]:24s} n={r['n']:5d}  cov95={r['cov95']:5.1f}  cov99={r['cov99']:5.1f}")

    # 中文注释：三联图同时输出新文件名和兼容旧文件名两份。
    print("  Plotting applicability-domain triptych ...")
    plot_applicability_domain(PC_train, PC_arch, high_mask, ad, cov_rows,
                              var1, var2,
                              paths=[OUT_ROOT / "pca_applicability_domain.png",
                                     OUT_ROOT / "pca_v1_overall.png"])

    cov_df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "group"},
                            "group": r["group"].replace("\n", " ")} for r in cov_rows])
    cov_df = cov_df[["group", "n", "cov95", "cov99"]]
    cov_df.insert(0, "n_pc", ad["n_pc"])
    cov_df["q95_dist"] = round(ad["q95"], 4)
    cov_df["q99_dist"] = round(ad["q99"], 4)
    coverage_csv = OUT_ROOT / "pca_applicability_coverage.csv"
    cov_df.to_csv(coverage_csv, index=False)
    print(f"    Saved: {coverage_csv}")

    print("  Plotting per-class panels ...")
    plot_per_class(PC_train, PC_arch, labels, classes, colors, high_mask,
                   var1, var2, "cfb6920_missing_mask", OUT_ROOT / "pca_v2_per_class.png")

    print("  Computing ellipse overlap ...")
    summary = compute_overlap(PC_train, PC_arch, labels, classes, high_mask, low_mask)
    summary.to_csv(OUT_ROOT / "pca_overlap_summary.csv", index=False)
    print(summary.to_string(index=False))

    np.savez(OUT_ROOT / "pca_coords.npz",
             PC_train=PC_train, PC_arch=PC_arch, train_labels=labels,
             arch_conf=arch_conf if arch_conf is not None else np.array([]),
             dist_modern=ad["dist_modern"], dist_arch=ad["dist_arch"],
             ad_q95=ad["q95"], ad_q99=ad["q99"], ad_n_pc=ad["n_pc"],
             explained_variance_ratio=pca.explained_variance_ratio_,
             components=pca.components_)

    loadings = pd.DataFrame(pca.components_[:2].T, columns=["PC1_loading", "PC2_loading"], index=ALL_ELEMENTS)
    loadings["PC1_abs"], loadings["PC2_abs"] = loadings["PC1_loading"].abs(), loadings["PC2_loading"].abs()
    loadings.sort_values("PC1_abs", ascending=False).to_csv(OUT_ROOT / "pca_loadings.csv")
    print(f"    Saved: pca_coords.npz, pca_loadings.csv, pca_overlap_summary.csv")

    return {
        "variant": "cfb6920_missing_mask", "output_dir": str(OUT_ROOT),
        "pc1_variance_pct": round(var1 * 100, 2),
        "pc2_variance_pct": round(var2 * 100, 2),
        "pc12_variance_pct": round((var1 + var2) * 100, 2),
        "ad_n_pc": ad["n_pc"], "ad_cum_var_pct": round(ad["cum_var"] * 100, 1),
        "ad_q95": round(ad["q95"], 4), "ad_q99": round(ad["q99"], 4),
        "modern_classes": len(classes), "modern_samples": int(len(PC_train)),
        "archean_samples": int(len(PC_arch)),
        "high_conf_samples": int(high_mask.sum()) if high_mask is not None else None,
        "low_conf_samples": int(low_mask.sum()) if low_mask is not None else None,
    }


# =========================
# 入口
# =========================
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Loading data ...")

    modern_raw = clean_columns(pd.read_csv(RAW_MODERN_PATH, low_memory=False))
    archean_raw = pd.read_csv(ARCHEAN_RAW_PATH, low_memory=False)
    print(f"  Modern raw : {len(modern_raw):,} rows")
    print(f"  Archean raw: {len(archean_raw):,} rows")

    for name, df in [("Modern raw", modern_raw), ("Archean raw", archean_raw)]:
        missing = [e for e in ALL_ELEMENTS if e not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    if archean_raw["AGE"].isna().any():
        raise ValueError("Age-constrained Archean input still contains missing AGE values")

    # 中文注释：正式PCA只使用无水SiO2 44-53 wt%的3012条太古代样品。
    archean_raw = preprocess_archean(archean_raw)
    if len(archean_raw) != 3012:
        raise ValueError(
            f"正式太古代PCA输入应为3012条，实际为 {len(archean_raw)} 条"
        )
    print(f"  Archean after 44-53 wt% filtering: {len(archean_raw):,} rows")

    # 标签清理
    modern_raw.dropna(subset=[TECTONIC_COL], inplace=True)
    modern_raw[TECTONIC_COL] = modern_raw[TECTONIC_COL].astype(str).str.strip().str.upper()
    modern_raw.reset_index(drop=True, inplace=True)
    # 中文注释：PCA现代参考集使用与最终GeoDAN完全一致的CFB确定性欠采样。
    cfb_indices = np.flatnonzero(
        modern_raw[TECTONIC_COL].to_numpy() == "CONTINENTAL FLOOD BASALT"
    )
    if len(cfb_indices) < CFB_TARGET_COUNT:
        raise ValueError(
            f"现代训练集CFB只有 {len(cfb_indices)} 条，不能保留 {CFB_TARGET_COUNT} 条"
        )
    selected_cfb = np.sort(
        np.random.default_rng(RANDOM_SEED).choice(
            cfb_indices,
            size=CFB_TARGET_COUNT,
            replace=False,
        )
    )
    non_cfb_indices = np.flatnonzero(
        modern_raw[TECTONIC_COL].to_numpy() != "CONTINENTAL FLOOD BASALT"
    )
    keep_indices = np.sort(np.concatenate([non_cfb_indices, selected_cfb]))
    modern_raw = modern_raw.iloc[keep_indices].reset_index(drop=True)
    print(f"  Modern raw after label cleanup: {len(modern_raw):,}")

    labels_raw = modern_raw[TECTONIC_COL].values
    classes = sorted(np.unique(labels_raw), key=lambda c: -(labels_raw == c).sum())
    print(f"\nClasses ({len(classes)}):")
    for cls in classes:
        print(f"  {abbr(cls):6s} <- {cls:32s} n_raw = {(labels_raw == cls).sum():,}")

    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(classes))))
    colors = {cls: palette[i] for i, cls in enumerate(classes)}

    # 中文注释：直接读取最终CFB=6920、Mask、不插补方案的预测概率，避免PCA另行推理。
    final_prediction = pd.read_csv(FINAL_PREDICTION_PATH, low_memory=False)
    if len(final_prediction) != len(archean_raw):
        raise ValueError(
            f"最终预测与太古代原始数据行数不一致: "
            f"{len(final_prediction)} != {len(archean_raw)}"
        )
    arch_conf = pd.to_numeric(
        final_prediction["Arc_probability3"],
        errors="coerce",
    ).to_numpy()
    if not np.isfinite(arch_conf).all():
        raise ValueError("Arc_probability3 contains missing or infinite values")
    high_mask = arch_conf >= HIGH_CONF_THRESHOLD
    low_mask = arch_conf < LOW_CONF_THRESHOLD
    print(f"\nArchean high arc-affinity by GeoDAN (P_arc >= {HIGH_CONF_THRESHOLD}): "
          f"{high_mask.sum():,} / {len(arch_conf):,}")
    print(f"Archean low  arc-affinity by GeoDAN (P_arc <  {LOW_CONF_THRESHOLD}): "
          f"{low_mask.sum():,} / {len(arch_conf):,}")

    # 中文注释：太古代连续数据保持原始缺失；PCA中仅将缺失放在现代标准化均值位置。
    modern_raw_feats = extract_elements(modern_raw)
    arch_raw_feats = extract_elements(archean_raw)

    # 中文注释：PCA 使用连续含量，不使用分位数分箱后的1-255编码。
    result = run_continuous_pca(
        modern_raw_feats,
        arch_raw_feats,
        labels_raw,
        classes,
        colors,
        high_mask,
        low_mask,
        arch_conf,
    )
    print(
        "\nContinuous-data PCA summary:\n"
        f"{pd.DataFrame([result]).to_string(index=False)}\n\nDone."
    )


if __name__ == "__main__":
    main()
