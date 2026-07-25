from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 标准库导入
# ──────────────────────────────────────────────────────────────────────────────
import sys
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 第三方库导入
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# === 统一路径配置：所有数据/模型路径来自 config/paths.py ===
import importlib.util as _importlib_util

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "06_archean_application"))
sys.path.insert(0, str(_PROJECT_ROOT))
from config.paths import (
    ARCHEAN_S3_CSV, ARCHEAN_DATA_SUBDIR, ARCHEAN_CASE_DIR,
    ARCHEAN_POOL_CSV, ARCHEAN_FINAL_DIR, ARCHEAN_FINAL_MASK_CSV,
    ARCHEAN_FINAL_PREDICTIONS_CSV,
    TRAIN_NORM_CSV, QUANTILE_PARAMS_JSON, MAIN_MODEL_WEIGHT,
)

# 中文注释：当前工程保留 archean_s3_preprocess.py 命名（CNNtest 中为 archean_data_preprocess.py）。
from archean_s3_preprocess import (
    CASE_STUDIES_ORDER,
    CaseStudyConfig,
    build_case_study_configs,
    load_final_age_constrained_pool,
    preprocess_archean,
    preprocess_case_study,
)

# 中文注释：04_model 目录以数字开头，无法用常规包导入，改用 importlib 加载训练脚本。
_TRAINING_MODULE_FILE = _PROJECT_ROOT / "04_model" / "ablation_v4_vit_transformer.py"
_TRAINING_MODULE_CACHE = None


def _load_training_module():
    """加载训练脚本模块，复用其中的正式列顺序与模型类，避免内联漂移。"""
    global _TRAINING_MODULE_CACHE
    if _TRAINING_MODULE_CACHE is None:
        spec = _importlib_util.spec_from_file_location(
            "ablation_v4_vit_transformer", str(_TRAINING_MODULE_FILE)
        )
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TRAINING_MODULE_CACHE = module
    return _TRAINING_MODULE_CACHE

# ──────────────────────────────────────────────────────────────────────────────
# SCI 绘图样式（按 v9 修订执行计划第六部分美化方案统一配置）
#   · 字体：Arial / Helvetica（v9 方案明确指定，sans-serif 系族）
#   · 标题 11–12 pt 粗体；坐标轴标签 9–10 pt；刻度 8–9 pt
#   · 统一输出 300 dpi PNG 位图
# ──────────────────────────────────────────────────────────────────────────────
SCI_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "legend.frameon":    True,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.7",
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.minor.size":  2,
    "ytick.minor.size":  2,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.45,
    "grid.alpha":        0.45,
    "grid.color":        "#bbbbbb",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "ps.fonttype":       42,
}
plt.rcParams.update(SCI_RC)


def _configure_chinese_font() -> None:
    """中文注释：注册 Windows 常见中文字体，避免中文标题和图注保存为方框。"""
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            current_fonts = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = [family_name] + current_fonts
            break


_configure_chinese_font()


# ──────────────────────────────────────────────────────────────────────────────
# 统一调色板（v9 修订执行计划 §6.2）
# ──────────────────────────────────────────────────────────────────────────────
COLOR_PRIMARY         = "#2E5C8A"   # 深蓝 — 主模型 / 9-class 全样品
COLOR_HIGHLIGHT       = "#D8624C"   # 橙红 — 关键结果 / 9-class 高置信
COLOR_SECONDARY       = "#A0937D"   # 暖灰 — 对比基准 (Liu 2024)
COLOR_TERTIARY        = "#7BA7BC"   # 浅蓝 — 次要项
COLOR_TARGET_LINE     = "#4DAC26"   # 绿色 — 参考线（如 target accuracy）
COLOR_TRANSITION_BAND = "#D9D9D9"   # 浅灰 — 3.2–2.5 Ga 板块构造过渡期背景带
COLOR_TEXT_MUTED      = "#555555"   # 深灰 — 注释文字


# ──────────────────────────────────────────────────────────────────────────────
# 太古代地质年代分期（IUGS, ICS 标准）
#   · Eoarchean    4.00–3.60 Ga
#   · Paleoarchean 3.60–3.20 Ga
#   · Mesoarchean  3.20–2.80 Ga
#   · Neoarchean   2.80–2.50 Ga
# 颜色按 ICS Stratigraphic Chart 中 Archean 分期惯用粉紫色系，由深至浅渐变
# ──────────────────────────────────────────────────────────────────────────────
ARCHEAN_PERIODS = [
    {"name": "Eoarchean",    "start_ga": 4.00, "end_ga": 3.60, "color": "#C76A9C"},
    {"name": "Paleoarchean", "start_ga": 3.60, "end_ga": 3.20, "color": "#DA9DBA"},
    {"name": "Mesoarchean",  "start_ga": 3.20, "end_ga": 2.80, "color": "#EAC1D2"},
    {"name": "Neoarchean",   "start_ga": 2.80, "end_ga": 2.50, "color": "#F4DDE5"},
]


def _add_archean_period_bands(
    ax: plt.Axes,
    *,
    band_height: float = 0.045,
    band_pad: float = 0.005,
    label_fontsize: float = 9.0,
    show_labels: bool = True,
) -> None:
    """
    在给定 Axes 顶部添加太古代分期颜色带（Eoarchean/Paleoarchean/Mesoarchean/Neoarchean）。

    color band 通过 axes-fraction 坐标绘制于绘图区上方，不占用 data area。
    上层调用需为 figure 的顶部预留足够空间（建议 ax.set_title 时增大 pad，并相应增大 figure 顶部）。

    参数
    ----
    ax            : 目标 axes（X 轴需为 Ga，方向：老→年轻 / left→right 反向）
    band_height   : 色带高度（axes 比例，相对 axes 高度）
    band_pad      : 色带与坐标轴顶部之间的留白
    label_fontsize: 时代名称文字大小
    show_labels   : 是否在色带内绘制时代名称
    """
    # 用 blended transform：x 用 data，y 用 axes fraction，
    # 这样色带横向自动跟随 X 轴 data 范围，纵向位于绘图区上方固定位置。
    trans = ax.get_xaxis_transform()
    y0 = 1.0 + band_pad
    y1 = y0 + band_height

    for period in ARCHEAN_PERIODS:
        x_lo = min(period["start_ga"], period["end_ga"])
        x_hi = max(period["start_ga"], period["end_ga"])
        rect = mpatches.Rectangle(
            (x_lo, y0), x_hi - x_lo, band_height,
            transform=trans,
            facecolor=period["color"], edgecolor="white",
            linewidth=0.6, clip_on=False, zorder=5,
        )
        ax.add_patch(rect)

        if show_labels:
            ax.text(
                (x_lo + x_hi) / 2.0, (y0 + y1) / 2.0,
                period["name"],
                transform=trans,
                ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold",
                color="#2A2A2A",
                clip_on=False, zorder=6,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  ★ 配置区：修改此处的路径和参数即可，无需命令行参数
# ══════════════════════════════════════════════════════════════════════════════

# 原始太古代玄武岩 CSV（Liu et al. 2024 原始数据，含 Arc_probability3 列）
SOURCE_S3_CSV_PATH = Path(str(ARCHEAN_S3_CSV))

# 训练集 CSV（用于读取类别名称顺序）：必须与当前权重训练时使用的 CSV 完全一致。
TRAIN_PATH = Path(str(TRAIN_NORM_CSV))

# 推理批次大小
BATCH_SIZE = 256

# 置信度阈值：softmax 最大值 >= HIGH_PROB 且 std <= HIGH_STD 才算"高置信度"
HIGH_PROB = 0.5
HIGH_STD  = 0.05

# 正文 5.3 节主图固定展示的高置信阈值，不受 HIGH_PROB 诊断配置影响。
MAIN_ARC_RATIO_THRESHOLD = 0.7

# 年龄分箱大小（单位：Ma，百万年）
BIN_SIZE_MYR = 200

# 板块构造过渡期（Brown et al., 2020），用于 plot_arc_ratio 灰色背景带
PT_TRANSITION_GA = (2.5, 3.2)

# ── 9 类补充可视化开关 ─────────────────────────────────────────────────────
PLOT_CLASS_KDE_RIDGELINE = True # KDE 山脊图 — 看不同构造环境随年龄变化的相对丰度
PLOT_CLASS_BUBBLE_MATRIX = True # 气泡矩阵 — 类别 × age bin 的稀疏可视化


# ──────────────────────────────────────────────────────────────────────────────
# 特征列定义（与 geodan_main_model.py 的 COLUMN_ORDER_SCHEME 保持一致）
# 可由环境变量 GEODAN_COLUMN_SCHEME 覆盖（geodan_main_model.py 训练后自动设置）
# ──────────────────────────────────────────────────────────────────────────────

_COLUMN_ORDER_SCHEME = "v1"

# v1: 矩阵分支 — 元素周期表顺序
_COLUMNS_IMG_V1 = [
    'NA2O(WT%)', 'MGO(WT%)',   'CR(PPM)',    'AL2O3(WT%)', 'SIO2(WT%)',  'P2O5(WT%)',
    'K2O(WT%)',  'CAO(WT%)',   'TIO2(WT%)',  'V(PPM)',     'MNO(WT%)',   'FEOT(WT%)',
    'RB(PPM)',   'SR(PPM)',    'Y(PPM)',     'NB(PPM)',    'CO(PPM)',    'NI(PPM)',
    'BA(PPM)',   'LA(PPM)',    'CE(PPM)',    'PR(PPM)',    'ND(PPM)',    'ZR(PPM)',
    'SM(PPM)',   'EU(PPM)',    'GD(PPM)',    'TB(PPM)',    'DY(PPM)',    'HO(PPM)',
    'TH(PPM)',   'ER(PPM)',    'YB(PPM)',    'LU(PPM)',    'HF(PPM)',    'TA(PPM)',
]

# v2: 矩阵分支 — 地化亲缘分组（6×6）
_COLUMNS_IMG_V2 = [
    'RB(PPM)',   'BA(PPM)',    'TH(PPM)',    'SR(PPM)',    'K2O(WT%)',   'NA2O(WT%)',
    'LA(PPM)',   'CE(PPM)',    'NB(PPM)',    'TA(PPM)',    'PR(PPM)',    'ND(PPM)',
    'SM(PPM)',   'EU(PPM)',    'ZR(PPM)',    'HF(PPM)',    'GD(PPM)',    'TB(PPM)',
    'DY(PPM)',   'HO(PPM)',    'ER(PPM)',    'YB(PPM)',    'LU(PPM)',    'Y(PPM)',
    'SIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'MGO(WT%)',   'CAO(WT%)',   'TIO2(WT%)',
    'CR(PPM)',   'NI(PPM)',    'CO(PPM)',    'V(PPM)',     'MNO(WT%)',   'P2O5(WT%)',
]

COLUMNS_TO_EXTRACT = _COLUMNS_IMG_V2 if _COLUMN_ORDER_SCHEME == 'v2' else _COLUMNS_IMG_V1

# 9类分类体系中属于弧相关的类别名称
ARC_RELATED_LABELS = {"Continental arc", "Intra-oceanic arc", "Island arc"}

# 9类颜色方案（与正文 Figure 1 / Figure 2 全球分布图保持一致；
#   弧相关 = 暖色（红/橙），洋相关 = 蓝/绿色，陆内 = 紫色，色盲友好）
CLASS_COLORS = {
    "Continental arc":          "#D7191C",
    "Intra-oceanic arc":        "#F46D43",
    "Island arc":               "#FDAE61",
    "BACK-ARC_BASIN":           "#4575B4",
    "SPREADING_CENTER":         "#91BFDB",
    "OCEANIC PLATEAU":          "#1A9641",
    "OCEAN ISLAND":             "#A6D96A",
    "CONTINENTAL FLOOD BASALT": "#762A83",
    "CONTINENTAL_RIFT":         "#C2A5CF",
}

# 9 类简写（与原始研究配图惯例一致：CA / IOA / IA / BAB / MOR / OP / OI / CF / CR）
CLASS_ABBREVS = {
    "Continental arc":          "CA",
    "Intra-oceanic arc":        "IOA",
    "Island arc":               "IA",
    "BACK-ARC_BASIN":           "BAB",
    "SPREADING_CENTER":         "MOR",
    "OCEANIC PLATEAU":          "OP",
    "OCEAN ISLAND":             "OI",
    "CONTINENTAL FLOOD BASALT": "CF",
    "CONTINENTAL_RIFT":         "CR",
}

# 案例 6 panel 横向柱状图专用低饱和配色，避免小图中颜色过于花哨。
CASE_BAR_COLORS = {
    "Continental arc":          "#B65A57",
    "Intra-oceanic arc":        "#C98261",
    "Island arc":               "#D6A15A",
    "BACK-ARC_BASIN":           "#5E7FA7",
    "SPREADING_CENTER":         "#8EABC0",
    "OCEANIC PLATEAU":          "#5F8F69",
    "OCEAN ISLAND":             "#9BAF72",
    "CONTINENTAL FLOOD BASALT": "#7E6A8E",
    "CONTINENTAL_RIFT":         "#A894B3",
}


# ══════════════════════════════════════════════════════════════════════════════
#  模型架构定义（完整内联，无需外部 .py 文件）
# ══════════════════════════════════════════════════════════════════════════════

# ── 序列分支输入列顺序（随 GEODAN_COLUMN_SCHEME 切换）──────────
_COLUMNS_SEQ_V1 = [
    'RB(PPM)',   'K2O(WT%)',  'BA(PPM)',    'SR(PPM)',    'CAO(WT%)',  'NA2O(WT%)',
    'LA(PPM)',   'Y(PPM)',    'MGO(WT%)',   'PR(PPM)',    'CE(PPM)',   'ER(PPM)',
    'HO(PPM)',   'ND(PPM)',   'SM(PPM)',    'DY(PPM)',    'LU(PPM)',   'TB(PPM)',
    'GD(PPM)',   'YB(PPM)',   'EU(PPM)',    'TH(PPM)',    'AL2O3(WT%)','HF(PPM)',
    'ZR(PPM)',   'TIO2(WT%)', 'MNO(WT%)',  'V(PPM)',     'NB(PPM)',   'CR(PPM)',
    'TA(PPM)',   'FEOT(WT%)', 'CO(PPM)',   'NI(PPM)',    'SIO2(WT%)', 'P2O5(WT%)',
]
_COLUMNS_SEQ_V2 = [
    'RB(PPM)',   'BA(PPM)',   'TH(PPM)',   'K2O(WT%)',  'NA2O(WT%)',
    'NB(PPM)',   'TA(PPM)',   'LA(PPM)',   'CE(PPM)',
    'PR(PPM)',   'SR(PPM)',   'P2O5(WT%)',
    'ND(PPM)',   'SM(PPM)',   'ZR(PPM)',   'HF(PPM)',   'EU(PPM)',
    'TIO2(WT%)', 'AL2O3(WT%)','GD(PPM)',  'TB(PPM)',   'DY(PPM)',
    'HO(PPM)',   'Y(PPM)',    'ER(PPM)',   'YB(PPM)',   'LU(PPM)',
    'CAO(WT%)',  'V(PPM)',    'MNO(WT%)',
    'FEOT(WT%)', 'MGO(WT%)', 'SIO2(WT%)', 'CR(PPM)',   'NI(PPM)',   'CO(PPM)',
]
COLUMNS_ELECTRODE_ORDER = _COLUMNS_SEQ_V2 if _COLUMN_ORDER_SCHEME == 'v2' else _COLUMNS_SEQ_V1

# 中文注释：v1正式预测严格复用训练脚本中的矩阵与序列列顺序。
if _COLUMN_ORDER_SCHEME == "v1":
    _training_module = _load_training_module()
    COLUMNS_TO_EXTRACT = list(_training_module.ORIGINAL_IMAGE_COLUMNS)
    COLUMNS_ELECTRODE_ORDER = list(_training_module.COLUMNS_ELECTRODE_ORDER_V1)


def reshape_to_image(X_2d: np.ndarray) -> np.ndarray:
    """将 (N, 36) 特征矩阵重塑为 CNN 输入格式 (N, 1, 6, 6)，归一化区间为 [0, 1]。"""
    return X_2d.reshape(-1, 1, 6, 6).astype(np.float32)


def _build_model(num_classes: int, use_missing_mask: bool = False):
    """直接实例化训练脚本中的正式模型类，避免内联参数与权重漂移。"""
    CurrentGeoDAN = _load_training_module().ViT_Transformer_DualStream

    return CurrentGeoDAN(
        num_classes=num_classes,
        use_missing_mask=use_missing_mask,
    )


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ── Patch Embedding ─────────────────────────────────────────────────────
    class PatchEmbedding(nn.Module):
        def __init__(self, in_channels: int, patch_size: int,
                     embed_dim: int, num_patches: int) -> None:
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        def forward(self, x):
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return x + self.pos_embed

    # ── Transformer Block ────────────────────────────────────────────────────
    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int,
                     ff_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                                   dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
            )
            self.norm1   = nn.LayerNorm(embed_dim)
            self.norm2   = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.ffn(x))
            return x

    # ── ViT-Transformer Dual-Stream (v4: CLS token + GAP 四路融合) ──────────
    class ViT_Transformer_DualStream(nn.Module):
        """
        主模型 v4：ViT + Seq Transformer 双流（无 CNN）。
          矩阵分支: 6×6 亲缘矩阵 → Patch Embed → CLS+GAP → ViT Encoder
          序列分支: 36 相容性序列 → Linear Embed → CLS+PE → Transformer Encoder
          融合:    vit_cls + vit_gap + seq_cls + seq_gap → MLP 分类头
        """
        def __init__(self, num_classes: int, input_size: int = 6, patch_size: int = 2,
                     embed_dim: int = 128, num_heads: int = 8,
                     transformer_layers: int = 2, ff_dim: int = 256,
                     dropout: float = 0.15) -> None:
            super().__init__()
            self.num_patches = (input_size // patch_size) ** 2   # 9 patches
            self.seq_len     = input_size * input_size           # 36 tokens
            self.embed_dim   = embed_dim

            # ── 矩阵分支: ViT ──
            self.patch_embed = PatchEmbedding(1, patch_size, embed_dim, self.num_patches)
            self.vit_cls     = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.vit_cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.vit_blocks  = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(transformer_layers)
            ])
            self.vit_norm    = nn.LayerNorm(embed_dim)

            # ── 序列分支: Transformer ──
            self.seq_proj       = nn.Linear(1, embed_dim)
            self.seq_norm       = nn.LayerNorm(embed_dim)
            self.seq_cls        = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.seq_pos_embed  = nn.Parameter(
                torch.randn(1, self.seq_len + 1, embed_dim) * 0.02)
            self.seq_blocks     = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(transformer_layers)
            ])
            self.seq_final_norm = nn.LayerNorm(embed_dim)

            # ── 融合分类头: 4 路特征 (vit_cls + vit_gap + seq_cls + seq_gap) ──
            head_in = embed_dim * 4
            self.fusion = nn.Sequential(
                nn.Linear(head_in, 192),
                nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(192, 96),
                nn.LayerNorm(96),  nn.GELU(), nn.Dropout(dropout),
                nn.Linear(96, num_classes),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            nn.init.trunc_normal_(self.vit_cls,     std=0.02)
            nn.init.trunc_normal_(self.seq_cls,     std=0.02)
            nn.init.trunc_normal_(self.vit_cls_pos, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x, x_seq):
            B = x.size(0)

            # ── 矩阵分支前向 ──
            vit_tokens = self.patch_embed(x)                         # (B, 9, D)
            vit_cls    = self.vit_cls.expand(B, -1, -1) + self.vit_cls_pos
            vit_tokens = torch.cat([vit_cls, vit_tokens], dim=1)     # (B, 10, D)
            for blk in self.vit_blocks:
                vit_tokens = blk(vit_tokens)
            vit_tokens  = self.vit_norm(vit_tokens)
            vit_cls_out = vit_tokens[:, 0]
            vit_gap_out = vit_tokens[:, 1:].mean(dim=1)

            # ── 序列分支前向 ──
            seq_tokens = self.seq_norm(self.seq_proj(x_seq))         # (B, 36, D)
            seq_cls    = self.seq_cls.expand(B, -1, -1)
            seq_tokens = torch.cat([seq_cls, seq_tokens], dim=1)     # (B, 37, D)
            seq_tokens = seq_tokens + self.seq_pos_embed
            for blk in self.seq_blocks:
                seq_tokens = blk(seq_tokens)
            seq_tokens   = self.seq_final_norm(seq_tokens)
            seq_cls_out  = seq_tokens[:, 0]
            seq_gap_out  = seq_tokens[:, 1:].mean(dim=1)

            # ── 四路融合 ──
            fused = torch.cat([vit_cls_out, vit_gap_out,
                               seq_cls_out, seq_gap_out], dim=1)
            return self.fusion(fused)

    _TORCH_AVAILABLE = True

except Exception as exc:
    _TORCH_AVAILABLE = False
    # 中文注释：部分环境中 torch 已安装但 DLL 初始化失败，也按不可用处理，便于复用已有预测结果绘图。
    print(f"[WARNING] PyTorch 不可用，模型推理相关功能不可用：{exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 数据读取
# ──────────────────────────────────────────────────────────────────────────────

def read_csv_fallback(path: Path) -> pd.DataFrame:
    """读取 CSV，优先 UTF-8，失败则回退到 ISO-8859-1。"""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")


def load_class_names(train_path: Path) -> list[str]:
    """
    从训练集 CSV 的 'TECTONIC SETTING' 列读取类别名称列表。
    使用 pd.factorize 保证顺序与训练时编码完全一致。
    """
    df_train = read_csv_fallback(train_path)
    _, unique = pd.factorize(df_train["TECTONIC SETTING"])
    return list(unique)


# ──────────────────────────────────────────────────────────────────────────────
# 模型推理（使用内联模型类，无需外部脚本）
# ──────────────────────────────────────────────────────────────────────────────

def build_model_inputs(
    normalized: pd.DataFrame,
    missing_mask: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 quantile 变换后的表格数据（0~255）转换为模型所需的两路输入：
      · x_img：reshape 成 (N, 1, 6, 6) CNN 图像张量，归一化到 [0, 1]
      · x_seq：按标准电极电势顺序排列的 (N, 36, 1) 序列张量，归一化到 [0, 1]
    """
    x_img_2d = normalized[COLUMNS_TO_EXTRACT].to_numpy(dtype=np.float32)
    x_seq_2d = normalized[COLUMNS_ELECTRODE_ORDER].to_numpy(dtype=np.float32)
    x_img    = reshape_to_image(x_img_2d / 255.0)
    x_seq    = (x_seq_2d / 255.0)[:, :, np.newaxis].astype(np.float32)

    if missing_mask is not None:
        image_mask_columns = [
            f"missing_mask__{column}" for column in COLUMNS_TO_EXTRACT
        ]
        sequence_mask_columns = [
            f"missing_mask__{column}" for column in COLUMNS_ELECTRODE_ORDER
        ]
        missing_columns = [
            column
            for column in set(image_mask_columns + sequence_mask_columns)
            if column not in missing_mask.columns
        ]
        if missing_columns:
            raise ValueError(
                f"太古代缺失mask缺少列: {sorted(missing_columns)}"
            )
        if len(missing_mask) != len(normalized):
            raise ValueError(
                f"太古代缺失mask行数不一致: "
                f"{len(missing_mask)} != {len(normalized)}"
            )

        x_img_mask = reshape_to_image(
            missing_mask[image_mask_columns].to_numpy(dtype=np.float32)
        )
        x_seq_mask = missing_mask[
            sequence_mask_columns
        ].to_numpy(dtype=np.float32)[:, :, np.newaxis]
        x_img = np.concatenate([x_img, x_img_mask], axis=1)
        x_seq = np.concatenate([x_seq, x_seq_mask], axis=2)
    return x_img, x_seq


def _load_weights(weight_path: Path, model, device):
    """加载权重，兼容三种保存格式（直接 state_dict / 'state_dict' key / 'model_state_dict' key）。"""
    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 预测列与统计汇总
# ──────────────────────────────────────────────────────────────────────────────

def add_prediction_columns(
    s3: pd.DataFrame,
    probs_mean: np.ndarray,
    probs_std: np.ndarray,
    class_names: list[str],
    high_prob: float,
    high_std: float,
) -> pd.DataFrame:
    """
    在 S3 DataFrame 上追加预测相关列（各类别概率、置信度分级等）。

    置信度分级规则：
      high   → pred_prob_max >= high_prob 且 pred_prob_std <= high_std
      medium → pred_prob_max >= 0.5
      low    → 其余
    """
    # 确保输入数组长度与 DataFrame 一致
    if len(probs_mean) != len(s3):
        raise ValueError(
            f"probs_mean length ({len(probs_mean)}) does not match s3 DataFrame length ({len(s3)})"
        )
    if len(probs_std) != len(s3):
        raise ValueError(
            f"probs_std length ({len(probs_std)}) does not match s3 DataFrame length ({len(s3)})"
        )
    
    out      = s3.copy()
    pred_idx = probs_mean.argmax(axis=1)
    max_prob = probs_mean.max(axis=1)
    pred_std = probs_std[np.arange(len(pred_idx)), pred_idx]

    out["pred_class_idx"]        = pred_idx
    out["pred_class_name"]       = [class_names[i] for i in pred_idx]
    out["pred_prob_max"]         = max_prob
    out["pred_prob_std"]         = pred_std
    out["is_arc_related_9class"] = out["pred_class_name"].isin(ARC_RELATED_LABELS)

    out["confidence_tier"] = np.select(
        [
            (max_prob >= high_prob) & (pred_std <= high_std),
            max_prob >= 0.5,
        ],
        ["high", "medium"],
        default="low",
    )
    out["is_high_confidence"] = out["confidence_tier"].eq("high")

    for i, name in enumerate(class_names):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        out[f"prob_{safe}"]     = probs_mean[:, i]
        out[f"prob_std_{safe}"] = probs_std[:, i]

    # 中文注释：公开数据中的弧概率统一使用最新模型三类弧环境概率之和。
    arc_indices = [
        index for index, name in enumerate(class_names)
        if name in ARC_RELATED_LABELS
    ]
    out["Arc_probability3"] = probs_mean[:, arc_indices].sum(axis=1)
    
    # Liu 2024 基线：优先使用原始数据中的 Liu2024_Arc_probability3（如果存在），
    # 否则回退到当前模型预测的 Arc_probability3
    if "Liu2024_Arc_probability3" in out.columns:
        liu_probability = pd.to_numeric(
            out["Liu2024_Arc_probability3"], errors="coerce"
        )
        # 中文注释：未匹配到 Liu 原始预测的样品必须保持缺失，不能当作非弧样品。
        out["liu_arc_binary"] = liu_probability.ge(0.5).where(
            liu_probability.notna()
        )
    else:
        print("      [警告] 未找到 Liu2024_Arc_probability3 列，liu_arc_binary 将使用当前模型预测")
        out["liu_arc_binary"] = out["Arc_probability3"].ge(0.5)

    return out


def _safe_ratio(num: int | float, den: int | float) -> float:
    """安全除法：分母为0时返回 NaN。"""
    return float(num) / float(den) if den else np.nan


def add_age_bins(df: pd.DataFrame, bin_size_myr: int) -> pd.DataFrame:
    """
    对 C_AGE（校正年龄，单位 Ma）列进行等宽分箱，新增 age_bin / age_bin_mid_ma / age_bin_label 列。
    """
    out   = df.copy()
    age = pd.to_numeric(out["C_AGE"], errors="coerce")
    # 中文注释：Liu优先使用校正年龄C_AGE；GeoROC没有C_AGE时使用补录的AGE。
    if "AGE" in out.columns:
        age = age.fillna(pd.to_numeric(out["AGE"], errors="coerce"))
    start = int(np.floor(age.min() / bin_size_myr) * bin_size_myr)
    stop  = int(np.ceil(age.max()  / bin_size_myr) * bin_size_myr + bin_size_myr)
    bins  = np.arange(start, stop + 1, bin_size_myr)
    out["age_bin"]        = pd.cut(age, bins=bins, include_lowest=True, right=False)
    out["age_bin_mid_ma"] = out["age_bin"].apply(
        lambda x: x.mid if pd.notna(x) else np.nan)
    out["age_bin_label"]  = out["age_bin"].apply(
        lambda x: f"{x.left/1000:.1f}–{x.right/1000:.1f} Ga" if pd.notna(x) else "")
    return out


def summarize_liu_baseline_by_age(
    source_path: Path,
    bin_size_myr: int,
) -> pd.DataFrame:
    """按正式44-53 wt%口径统计Liu原始Arc_probability3年龄曲线。"""
    liu = read_csv_fallback(source_path)
    # 中文注释：正式太古代口径固定为无水SiO2 44-53 wt%、MgO不大于18 wt%。
    liu = preprocess_archean(liu, expected_sample_count=2116)
    required_columns = {"C_AGE", "Arc_probability3"}
    missing_columns = required_columns.difference(liu.columns)
    if missing_columns:
        raise ValueError(
            f"Liu 原始数据缺少必要列: {sorted(missing_columns)}"
        )

    liu_direct = pd.DataFrame({
        "C_AGE": pd.to_numeric(liu["C_AGE"], errors="coerce"),
        "Liu2024_Arc_probability3": pd.to_numeric(
            liu["Arc_probability3"], errors="coerce"
        ),
    }).dropna(subset=["C_AGE", "Liu2024_Arc_probability3"])
    liu_direct = add_age_bins(liu_direct, bin_size_myr)

    rows = []
    for interval, group in liu_direct.dropna(subset=["age_bin"]).groupby(
        "age_bin", observed=True
    ):
        arc_count = int(group["Liu2024_Arc_probability3"].ge(0.5).sum())
        rows.append({
            "age_mid_ga": interval.mid / 1000.0,
            "n_liu_samples": len(group),
            "n_liu_arc": arc_count,
            "ratio_liu_arc": _safe_ratio(arc_count, len(group)),
        })

    print(
        f"      Liu 原始曲线直接读取: {source_path}，"
        f"有效样品 {len(liu_direct)}"
    )
    return pd.DataFrame(rows).sort_values("age_mid_ga", ascending=False)


def summarize_by_age(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """按年龄分箱统计各时段的样品数量、弧相关比例、Liu 2024 对比等指标，降序排列。"""
    rows = []
    for interval, group in df.dropna(subset=["age_bin"]).groupby("age_bin", observed=True):
        high = group[group["is_high_confidence"]]
        
        # Liu 2024 基线统计（只统计有 Liu 预测值的样品）
        liu_valid = group[group["liu_arc_binary"].notna()]
        n_liu_arc = int(liu_valid["liu_arc_binary"].sum()) if len(liu_valid) > 0 else 0
        ratio_liu_arc = _safe_ratio(n_liu_arc, len(liu_valid))
        
        row  = {
            "age_bin":               f"{interval.left:.0f}–{interval.right:.0f} Ma",
            "age_mid_ga":            interval.mid / 1000.0,
            "n_all":                 len(group),
            "n_high":                len(high),
            "n_9class_arc_all":      int(group["is_arc_related_9class"].sum()),
            "ratio_9class_arc_all":  _safe_ratio(group["is_arc_related_9class"].sum(), len(group)),
            "mean_arc_probability":  pd.to_numeric(
                group["Arc_probability3"],
                errors="coerce",
            ).mean(),
            "ratio_arc_probability_ge_0_5": pd.to_numeric(
                group["Arc_probability3"],
                errors="coerce",
            ).ge(0.5).mean(),
            "n_9class_arc_high":     int(high["is_arc_related_9class"].sum()),
            "ratio_9class_arc_high": _safe_ratio(high["is_arc_related_9class"].sum(), len(high)),
            "n_liu_samples":         len(liu_valid),
            "n_liu_arc":             n_liu_arc,
            "ratio_liu_arc":         ratio_liu_arc,
            "mean_pred_prob":        group["pred_prob_max"].mean(),
        }
        for name in class_names:
            row[f"n_{name}"]     = int(group["pred_class_name"].eq(name).sum())
            row[f"ratio_{name}"] = _safe_ratio(group["pred_class_name"].eq(name).sum(), len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("age_mid_ga", ascending=False)


def summarize_cratons(df: pd.DataFrame) -> pd.DataFrame:
    """按克拉通统计样品数、高置信弧相关样品数及最老/最新年龄，从早到晚排序。"""
    rows = []
    for craton, group in df.groupby("Craton", dropna=False):
        high_arc = group[group["is_high_confidence"] & group["is_arc_related_9class"]]
        liu_arc  = group[group["liu_arc_binary"]]
        rows.append({
            "Craton":                           craton,
            "n_all":                            len(group),
            "n_high":                           int(group["is_high_confidence"].sum()),
            "n_9class_arc_high":                len(high_arc),
            "oldest_9class_arc_high_C_AGE_Ma":  high_arc["C_AGE"].max() if len(high_arc) else np.nan,
            "youngest_9class_arc_high_C_AGE_Ma":high_arc["C_AGE"].min() if len(high_arc) else np.nan,
            "n_liu_arc":                        len(liu_arc),
            "oldest_liu_arc_C_AGE_Ma":          liu_arc["C_AGE"].max() if len(liu_arc) else np.nan,
            "dominant_high_arc_class":          high_arc["pred_class_name"].mode().iat[0]
                                                if len(high_arc) else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["oldest_9class_arc_high_C_AGE_Ma", "n_9class_arc_high"],
        ascending=[False, False],
    )


def summarize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算传统地球化学弧判别指标（Ba/Th、Th/Nb、Ba/La），
    按年龄分箱分别统计"所有样品"和"高置信弧相关样品"两个子集的均值/中位数。
    """
    work = df.copy()
    for col in ["NB", "LA", "TH", "BA"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["Ba_Th"] = work["BA"] / work["TH"].replace(0, np.nan)
    work["Th_Nb"] = work["TH"] / work["NB"].replace(0, np.nan)
    work["Ba_La"] = work["BA"] / work["LA"].replace(0, np.nan)

    rows = []
    for interval, group in work.dropna(subset=["age_bin"]).groupby("age_bin", observed=True):
        high_arc = group[group["is_high_confidence"] & group["is_arc_related_9class"]]
        for subset_name, subset in [("all", group), ("high_9class_arc", high_arc)]:
            rows.append({
                "age_bin":      f"{interval.left:.0f}–{interval.right:.0f} Ma",
                "age_mid_ga":   interval.mid / 1000.0,
                "subset":       subset_name,
                "n":            len(subset),
                "mean_Ba_Th":   subset["Ba_Th"].mean(),
                "median_Ba_Th": subset["Ba_Th"].median(),
                "mean_Th_Nb":   subset["Th_Nb"].mean(),
                "median_Th_Nb": subset["Th_Nb"].median(),
                "mean_Ba_La":   subset["Ba_La"].mean(),
                "median_Ba_La": subset["Ba_La"].median(),
            })
    return pd.DataFrame(rows).sort_values(["age_mid_ga", "subset"], ascending=[False, True])


# ──────────────────────────────────────────────────────────────────────────────
# 绘图辅助
# ──────────────────────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    """
    统一的保存接口：只输出 300 dpi PNG 文件。
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600)


def _add_panel_label(ax: plt.Axes, label: str,
                     x: float = -0.08, y: float = 0.97,
                     fontsize: int = 11) -> None:
    """
    在子图左上角内部添加 (a)/(b)/(c) 编号（粗体），符合 v9 美化方案规范。
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="top", ha="left", zorder=10)


# ──────────────────────────────────────────────────────────────────────────────
# Coverage-Accuracy 曲线（确定置信度阈值）
# ──────────────────────────────────────────────────────────────────────────────

def plot_coverage_accuracy(
    test_path: Path,
    class_names: list[str],
    config: RunConfig,
) -> Optional[float]:
    """
    在已标注的测试集上计算 Coverage-Accuracy 曲线，绘图并返回推荐的置信度阈值。

    Coverage：模型置信度 >= 阈值时保留的样品比例
    Accuracy：保留样品中预测正确的比例

    推荐阈值：保持 Accuracy >= 90%（TARGET_ACC）条件下的最高可用阈值。
    测试集 CSV 格式要求：包含 COLUMNS_TO_EXTRACT 特征列（已 quantile 变换，0~255）
                         及 'TECTONIC SETTING' 真实标签列。
    """
    if not _TORCH_AVAILABLE:
        print("[WARNING] 跳过 Coverage-Accuracy 曲线（PyTorch 不可用）。")
        return None

    from torch.utils.data import DataLoader, TensorDataset

    print(f"      读取测试集: {test_path}")
    df_test   = read_csv_fallback(test_path)
    name2idx  = {name: i for i, name in enumerate(class_names)}
    y_true    = df_test["TECTONIC SETTING"].map(name2idx).to_numpy(dtype=np.int64)
    valid_mask = ~np.isnan(y_true.astype(float))
    df_test   = df_test.iloc[np.where(valid_mask)[0]].reset_index(drop=True)
    y_true    = y_true[valid_mask]

    # 构建模型输入（测试集已是 quantile 变换后的 0~255 值）
    x_img_2d = df_test[COLUMNS_TO_EXTRACT].to_numpy(dtype=np.float32)
    x_seq_2d = df_test[COLUMNS_ELECTRODE_ORDER].to_numpy(dtype=np.float32)
    x_img    = reshape_to_image(x_img_2d / 255.0)
    x_seq    = (x_seq_2d / 255.0)[:, :, np.newaxis].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_img), torch.FloatTensor(x_seq)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    all_weight_probs: list[np.ndarray] = []
    for weight_path in config.weights:
        model = _build_model(len(class_names)).to(device)
        model = _load_weights(weight_path, model, device)
        model.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for b_img, b_seq in loader:
                logits = model(b_img.to(device), b_seq.to(device))
                parts.append(F.softmax(logits, dim=1).cpu().numpy())
        all_weight_probs.append(np.concatenate(parts, axis=0))

    probs_mean = np.mean(all_weight_probs, axis=0)
    max_probs  = probs_mean.max(axis=1)
    pred_cls   = probs_mean.argmax(axis=1)
    correct    = (pred_cls == y_true).astype(float)

    # 计算不同阈值下的 Coverage 和 Accuracy
    thresholds = np.linspace(0.0, 0.999, 400)
    coverages  = np.empty(len(thresholds))
    accuracies = np.full(len(thresholds), np.nan)
    MIN_SAMPLES = max(10, int(0.01 * len(max_probs)))  # 至少1%样品才计算accuracy

    for i, t in enumerate(thresholds):
        mask           = max_probs >= t
        coverages[i]   = mask.sum() / len(max_probs)
        if mask.sum() >= MIN_SAMPLES:
            accuracies[i] = correct[mask].mean()

    # 确定推荐阈值：accuracy >= TARGET_ACC 条件下的最高阈值
    TARGET_ACC  = 0.90
    valid_pts   = (~np.isnan(accuracies)) & (coverages >= 0.05) & (accuracies >= TARGET_ACC)
    if valid_pts.any():
        rec_thresh = float(thresholds[valid_pts][-1])
    else:
        # 降级：取 accuracy 最高点
        rec_thresh = float(thresholds[np.nanargmax(accuracies)])

    rec_coverage = float(coverages[np.argmin(np.abs(thresholds - rec_thresh))])
    rec_accuracy = float(accuracies[np.argmin(np.abs(thresholds - rec_thresh))])

    print(f"      推荐置信度阈值: {rec_thresh:.3f}  "
          f"→ Coverage: {rec_coverage:.1%}, Accuracy: {rec_accuracy:.1%}")

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    # 双栏宽度 ≈ 17 cm，对应 figsize ≈ (6.7, 4.4) inches
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # ── (a) Coverage-Accuracy 曲线 ────────────────────────────────────────────
    ax = axes[0]
    cov_plot = coverages[~np.isnan(accuracies)]
    acc_plot = accuracies[~np.isnan(accuracies)]
    sort_idx = np.argsort(cov_plot)

    ax.plot(cov_plot[sort_idx], acc_plot[sort_idx],
            color=COLOR_PRIMARY, lw=1.8, zorder=3,
            label="Coverage–Accuracy")
    ax.axhline(TARGET_ACC, color=COLOR_TARGET_LINE, lw=1.0, ls="--", alpha=0.9,
               label=f"Target accuracy ({TARGET_ACC:.0%})")
    ax.scatter([rec_coverage], [rec_accuracy],
               s=90, color=COLOR_HIGHLIGHT, zorder=5,
               edgecolors="white", lw=1.0,
               label=f"Recommended\n(cov = {rec_coverage:.2f}, acc = {rec_accuracy:.2f})")
    ax.axvline(rec_coverage, color=COLOR_HIGHLIGHT, lw=0.9, ls=":", alpha=0.7)
    ax.axhline(rec_accuracy, color=COLOR_HIGHLIGHT, lw=0.9, ls=":", alpha=0.7)

    ax.set_xlabel("Coverage (fraction of retained samples)")
    ax.set_ylabel("Accuracy on retained samples")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(min(0.5, np.nanmin(acc_plot) - 0.05), 1.02)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right")
    _add_panel_label(ax, "(a)")

    # ── (b) 阈值 vs. Coverage / Accuracy ──────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(thresholds, coverages, color=COLOR_PRIMARY, lw=1.8, zorder=3,
             label="Coverage vs. threshold")
    ax2.axvline(rec_thresh, color=COLOR_HIGHLIGHT, lw=1.2, ls="--", zorder=4,
                label=f"Threshold = {rec_thresh:.3f}")
    ax2.axhline(rec_coverage, color=COLOR_HIGHLIGHT, lw=0.9, ls=":", alpha=0.7)
    ax2.scatter([rec_thresh], [rec_coverage],
                s=90, color=COLOR_HIGHLIGHT, zorder=5,
                edgecolors="white", lw=1.0)

    # 次坐标轴叠加 accuracy
    ax2r = ax2.twinx()
    ax2r.plot(thresholds, accuracies, color=COLOR_SECONDARY, lw=1.3, ls="-.",
              alpha=0.85, zorder=2, label="Accuracy")
    ax2r.set_ylabel("Accuracy on retained samples", color=COLOR_SECONDARY)
    ax2r.tick_params(axis="y", labelcolor=COLOR_SECONDARY)
    ax2r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2r.set_ylim(0.5, 1.02)
    ax2r.spines["top"].set_visible(False)
    ax2r.grid(False)

    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("Coverage (fraction of retained samples)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # 合并双坐标轴图例
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")
    _add_panel_label(ax2, "(b)")

    fig.suptitle(
        f"Coverage–Accuracy analysis  |  recommended threshold = {rec_thresh:.3f}",
        fontsize=11, y=1.02, fontweight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, config.fig_coverage_accuracy_path)
    plt.close(fig)
    return rec_thresh


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：弧相关比例随年龄变化（Figure 11，全文核心图）
# ──────────────────────────────────────────────────────────────────────────────

# 三条曲线统一样式（v9 §6.2 特别提示：不同 marker + 不同线型）
_ARC_LINE_STYLES = [
    # 9-class all：主色深蓝 + 圆 + 实线
    {"color": COLOR_PRIMARY,   "marker": "o", "ms": 6.0, "lw": 1.6, "ls": "-"},
    # 9-class high-confidence：突出色橙红 + 方 + 短划线
    {"color": COLOR_HIGHLIGHT, "marker": "s", "ms": 6.0, "lw": 1.6, "ls": "--"},
    # Liu 2024：辅色暖灰 + 三角 + 点线
    {"color": COLOR_SECONDARY, "marker": "^", "ms": 6.5, "lw": 1.6, "ls": ":"},
]


def plot_arc_ratio(
    df: pd.DataFrame,
    probs_mean: np.ndarray,
    age_summary: pd.DataFrame,
    liu_age_summary: pd.DataFrame,
    output_path: Path,
    high_threshold: float = MAIN_ARC_RATIO_THRESHOLD,
    high_std: float = HIGH_STD,
) -> None:
    """
    绘制正文 5.3 节弧相关 affinity 随年龄变化的主图（all samples + ≥0.7 + Liu 2024 ≥0.5）。

    特征：
      · 三条曲线分别用不同 marker（圆/方/三角）和不同线型（实/虚/点），便于黑白印刷区分；
      · 在 3.2–2.5 Ga 区间叠加灰色背景带，标注 "Plate tectonic transition (Brown et al., 2020)"；
      · 在标题下方加入太古代分期色带（Eoarchean/Paleoarchean/Mesoarchean/Neoarchean）；
      · X 轴反向，老年代在左、年轻在右。
    """
    # 适当增大 figure 高度，以容纳标题下方的 Archean 色带
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    # 中文说明：这张图沿用旧版展示口径，用每个 200 Myr 分箱的年轻端作横坐标。
    # 例如 3200–3400 Ma 画在 3.2 Ga，而不是分箱中点 3.3 Ga。
    plot_age_ga = age_summary["age_mid_ga"].to_numpy(dtype=float)   # 不再 -0.1
    visible_mask = plot_age_ga >= 2.5                                # 这样 mid=2.5 也保留
    plot_summary = age_summary.loc[visible_mask].reset_index(drop=True)
    x = plot_age_ga[visible_mask]
    threshold_summary = _compute_arc_ratio_by_threshold(
        df, probs_mean, high_threshold, high_std,
    )
    threshold_visible = threshold_summary["age_mid_ga"] >= 2.5
    x_high = threshold_summary.loc[threshold_visible, "age_mid_ga"].to_numpy(dtype=float)
    y_high = threshold_summary.loc[threshold_visible, "ratio_arc"].to_numpy(dtype=float)
    liu_visible = liu_age_summary["age_mid_ga"] >= 2.5
    x_liu = liu_age_summary.loc[liu_visible, "age_mid_ga"].to_numpy(dtype=float)
    y_liu = liu_age_summary.loc[liu_visible, "ratio_liu_arc"].to_numpy(dtype=float)

    # ── 1. 先画板块构造过渡期背景带（zorder=0，置于所有曲线下方） ────────────
    band_low, band_high = PT_TRANSITION_GA
    ax.axvspan(band_low, band_high,
               color=COLOR_TRANSITION_BAND, alpha=0.55, zorder=0,
               label="_nolegend_")
    # 在带内顶部标注（用 axes 坐标系的 y、data 坐标系的 x）
    ax.text(
        (band_low + band_high) / 2.0, 0.96,
        "Plate tectonic transition\n(Brown et al., 2020)",
        ha="center", va="top",
        fontsize=8, color=COLOR_TEXT_MUTED, style="italic",
        transform=ax.get_xaxis_transform(),
        zorder=1,
    )

    # ── 2. 三条曲线 ──────────────────────────────────────────────────────────
    data_series = [
        (x,
         plot_summary["mean_arc_probability"].to_numpy(dtype=float),
         "Mean arc probability",
         _ARC_LINE_STYLES[0]),
        (x_high,
         y_high,
         f"Samples with P$_{{arc}}$ ≥ {high_threshold:.1f}",
         _ARC_LINE_STYLES[1]),
        (x_liu,
         y_liu,
         "Liu et al., 2024 ( ≥ 0.5)",
         _ARC_LINE_STYLES[2]),
    ]
    for x_vals, y_vals, label, sty in data_series:
        ax.plot(
            x_vals, y_vals,
            color=sty["color"], marker=sty["marker"],
            markersize=sty["ms"], linewidth=sty["lw"], linestyle=sty["ls"],
            markerfacecolor="white", markeredgewidth=1.3,
            zorder=3, label=label,
        )

    # ── 3. 坐标轴与图例 ──────────────────────────────────────────────────────
    ax.set_xlabel("Age (Ga)")
    ax.set_ylabel("Arc-related affinity")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # 固定 X 轴为完整太古代区间 (4.0–2.5 Ga)，使色块完整覆盖且与数据对齐
    ax.set_xlim(4.00, 2.48)        # 反向：老在左、年轻在右
    ax.set_xticks(np.arange(2.6, 4.01, 0.2))

    # 图例放左上：避开右上 3.2–2.5 Ga 过渡带的 Brown et al. (2020) 标注
    ax.legend(loc="upper left", framealpha=0.92)

    # ── 4. 标题 + 标题下方 Archean 时代色带 ─────────────────────────────────
    # 增大 pad 为 Archean 色带预留空间
    # ax.set_title(
    #     "Arc-related affinity through geological time",
    #     loc="left", pad=28,
    # )
    _add_archean_period_bands(
        ax,
        band_height=0.055,
        band_pad=0.010,
        label_fontsize=9.5,
    )

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：置信度阈值敏感性分析（0.5 / 0.6 / 0.7 / 0.8 四条阈值曲线）
# ──────────────────────────────────────────────────────────────────────────────

# 敏感性分析的阈值列表
SENSITIVITY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

# 阈值曲线配色：0.7 对应正文主图阈值，0.8 用更强橙红强调严格筛选。
_SENS_COLORS = [
    "#4575B4",  # 阈值 0.5: 经典蓝（源自 BAB 配色，冷色起点，代表最宽松条件）
    "#9BAF72",  # 阈值 0.6: 柔和绿（源自案例图的低饱和绿，冷暖过渡）
    "#FDAE61",  # 阈值 0.7: 正文主图阈值，用暖黄/亮橙引导视觉重心
    "#D8624C",  # 阈值 0.8: 更严格筛选，用橙红强调高门槛结果
]
_SENS_LS     = ["-",       "-.",      "--",      ":"]


def _compute_arc_ratio_by_threshold(
    df: pd.DataFrame,
    probs_mean: np.ndarray,
    threshold: float,
    high_std: float = HIGH_STD,
) -> pd.DataFrame:
    """
    统计每个年龄分箱中 Arc_probability3 达到阈值的样品比例。

    中文注释：分母始终是该年龄箱全部样品，阈值作用于三类弧概率之和，
    与“高弧样品 Arc_probability3 >= threshold”的汇总口径完全一致。
    """
    sub = df.copy()
    arc_probability = pd.to_numeric(
        sub["Arc_probability3"],
        errors="coerce",
    )
    sub["_above_thresh"] = arc_probability.ge(threshold)

    rows = []
    for interval, group in sub.dropna(subset=["age_bin"]).groupby("age_bin", observed=True):
        high_count = int(group["_above_thresh"].sum())
        ratio = _safe_ratio(high_count, len(group))
        rows.append({
            "age_mid_ga": interval.mid / 1000.0,
            "ratio_arc":  ratio,
            "n_high":     high_count,
        })
    return pd.DataFrame(rows).sort_values("age_mid_ga")


def plot_arc_ratio_sensitivity(
    df: pd.DataFrame,
    probs_mean: np.ndarray,
    output_path: Path,
    high_std: float = HIGH_STD,
) -> None:
    """
    置信度阈值敏感性分析图：在 0.5 / 0.6 / 0.7 / 0.8 四个阈值下分别画出
    弧相关高置信样品比例的时间演化曲线。
    所有曲线画在同一个坐标系，便于评估阈值选择对结论稳健性的影响。
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    # ── 1. 板块构造过渡期背景带 ───────────────────────────────────────────────
    band_low, band_high = PT_TRANSITION_GA
    ax.axvspan(band_low, band_high,
               color=COLOR_TRANSITION_BAND, alpha=0.55, zorder=0,
               label="_nolegend_")
    ax.text(
        (band_low + band_high) / 2.0, 0.96,
        "Plate tectonic transition\n(Brown et al., 2020)",
        ha="center", va="top",
        fontsize=8, color=COLOR_TEXT_MUTED, style="italic",
        transform=ax.get_xaxis_transform(),
        zorder=1,
    )

    # ── 2. 四个阈值曲线 ──────────────────────────────────────────────────────
    for thresh, color, ls in zip(SENSITIVITY_THRESHOLDS, _SENS_COLORS, _SENS_LS):
        sens_df = _compute_arc_ratio_by_threshold(df, probs_mean, thresh, high_std)
        visible = sens_df["age_mid_ga"] >= 2.5
        x_s = sens_df.loc[visible, "age_mid_ga"].to_numpy(dtype=float)
        y_s = sens_df.loc[visible, "ratio_arc"].to_numpy(dtype=float)
        lw  = 2.0 if thresh == MAIN_ARC_RATIO_THRESHOLD else 1.4
        ax.plot(
            x_s, y_s,
            color=color, linestyle=ls, linewidth=lw,
            marker="o", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.2,
            zorder=3,
            label=f"≥ {thresh:.1f}",
        )

    # ── 3. 坐标轴与图例 ──────────────────────────────────────────────────────
    ax.set_xlabel("Age (Ga)")
    ax.set_ylabel("Arc-related affinity (high-confidence samples)")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(4.00, 2.48)
    ax.set_xticks(np.arange(2.6, 4.01, 0.2))
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8)

    # ── 4. 标题 + 太古代分期色带 ─────────────────────────────────────────────
    ax.set_title(
        "Confidence threshold sensitivity — arc-related affinity through time",
        loc="left", pad=28,
    )
    _add_archean_period_bands(
        ax, band_height=0.055, band_pad=0.010, label_fontsize=9.5,
    )

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：太古代玄武岩 arc-related affinity 全球空间分布图
# ──────────────────────────────────────────────────────────────────────────────

def _arc_affinity_from_probabilities(
    probs_mean: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """从 9 类概率矩阵中提取三类弧相关类别的概率和，作为 arc-related affinity。"""
    arc_indices = [i for i, name in enumerate(class_names) if name in ARC_RELATED_LABELS]
    if not arc_indices:
        return np.full(probs_mean.shape[0], np.nan, dtype=float)
    return probs_mean[:, arc_indices].sum(axis=1)


def _draw_schematic_world_land(ax: plt.Axes) -> None:
    """
    绘制内置简化全球陆地轮廓。

    当前运行环境没有 cartopy/geopandas/shapefile 数据读取能力，因此这里用低细节多边形
    做轻量底图，只提供全球空间参照，不参与任何定量分析。
    """
    land_polygons = [
        # North America
        [(-168, 15), (-160, 55), (-135, 70), (-95, 72), (-55, 54),
         (-62, 35), (-82, 25), (-105, 15), (-125, 24), (-145, 20)],
        # Central America
        [(-116, 15), (-96, 20), (-78, 10), (-84, 7), (-99, 13)],
        # South America
        [(-81, 12), (-64, 9), (-48, -6), (-35, -24), (-53, -55),
         (-72, -55), (-80, -20)],
        # Greenland
        [(-73, 60), (-55, 82), (-25, 78), (-20, 62), (-45, 58)],
        # Europe and Asia
        [(-11, 35), (-9, 58), (24, 71), (62, 70), (110, 72),
         (160, 62), (178, 52), (150, 28), (112, 12), (78, 8),
         (47, 22), (28, 36), (10, 36)],
        # Africa
        [(-18, 35), (10, 37), (34, 31), (50, 12), (43, -30),
         (25, -35), (7, -33), (-10, -22), (-17, 5)],
        # Arabia
        [(36, 30), (56, 25), (58, 14), (47, 11), (39, 18)],
        # India
        [(68, 24), (88, 22), (82, 8), (76, 6), (70, 15)],
        # Southeast Asia
        [(96, 20), (118, 18), (126, 5), (108, -6), (96, 6)],
        # Australia
        [(112, -11), (154, -12), (154, -38), (132, -44), (113, -32)],
        # Madagascar
        [(47, -13), (51, -25), (46, -26), (43, -18)],
        # Antarctica, cropped by y-limit but retained as bottom reference.
        [(-180, -60), (-120, -70), (-60, -66), (0, -72),
         (60, -66), (120, -70), (180, -60)],
    ]

    for polygon in land_polygons:
        patch = mpatches.Polygon(
            polygon,
            closed=True,
            facecolor="#E3E6DE",
            edgecolor="#B9BFB4",
            linewidth=0.65,
            zorder=1.5,
        )
        ax.add_patch(patch)


def _try_draw_basemap_world(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[Optional[object], np.ndarray, np.ndarray]:
    """
    优先复用 data_analysis/distribution.py 的 Basemap 世界地图画法。

    返回 Basemap 对象和投影后的坐标；若当前 Python 环境没有 Basemap，则返回 None 与原始经纬度。
    """
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError:
        return None, lon, lat

    basemap = Basemap(
        projection="cyl",
        resolution="c",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        ax=ax,
    )
    # 中文注释：稳定画出真正的世界地图底图，不依赖 bluemarble 图片资源是否可用。
    basemap.drawmapboundary(fill_color="#DCEBF2", linewidth=0.6)
    basemap.fillcontinents(color="#F1EDE3", lake_color="#DCEBF2", zorder=1)
    basemap.drawcoastlines(color="#6F756F", linewidth=0.45, zorder=2)
    basemap.drawcountries(color="#A5ABA3", linewidth=0.25, zorder=2)
    x_map, y_map = basemap(lon, lat)
    return basemap, np.asarray(x_map), np.asarray(y_map)


def plot_arc_affinity_global_map(
    df: pd.DataFrame,
    probs_mean: np.ndarray,
    class_names: list[str],
    output_path: Path,
    high_threshold: float = MAIN_ARC_RATIO_THRESHOLD,
    high_std: float = HIGH_STD,
) -> None:
    """
    绘制太古代玄武岩判别结果的 arc-related affinity 全球空间分布。

    采用纯 matplotlib 的简约经纬网底图，避免依赖 cartopy/geopandas。
    点颜色为 Continental arc + Intra-oceanic arc + Island arc 三类概率和；
    黑色描边点表示在正文阈值下的高置信弧相关样品。
    """
    work = df.copy()
    work["_lat"] = pd.to_numeric(work["LATITUDE"], errors="coerce")
    work["_lon"] = pd.to_numeric(work["LONGITUDE"], errors="coerce")
    # 中文注释：把 0–360 经度统一转换到 -180–180，保证全球图不会在太平洋边界断开。
    work["_lon"] = ((work["_lon"] + 180.0) % 360.0) - 180.0
    work["_arc_affinity"] = _arc_affinity_from_probabilities(probs_mean, class_names)

    valid = (
        work["_lat"].between(-90.0, 90.0)
        & work["_lon"].between(-180.0, 180.0)
        & work["_arc_affinity"].notna()
    )
    plot_df = work.loc[valid].copy()
    if plot_df.empty:
        print("      [跳过] arc affinity 全球图：没有有效经纬度样品。")
        return

    max_prob = probs_mean[valid.to_numpy()].max(axis=1)
    pred_std = pd.to_numeric(plot_df["pred_prob_std"], errors="coerce").to_numpy(dtype=float)
    high_arc = (
        plot_df["is_arc_related_9class"].to_numpy(dtype=bool)
        & (max_prob >= high_threshold)
        & (pred_std <= high_std)
    )

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    lon_values = plot_df["_lon"].to_numpy(dtype=float)
    lat_values = plot_df["_lat"].to_numpy(dtype=float)
    basemap, x_values, y_values = _try_draw_basemap_world(ax, lon_values, lat_values)

    if basemap is None:
        ax.set_facecolor("#F4F7FA")
        # 中文注释：没有 Basemap 时退回简化陆地轮廓，保证图仍有全球空间参照。
        for lat0 in range(-60, 90, 30):
            if (lat0 // 30) % 2 == 0:
                ax.axhspan(lat0, lat0 + 30, color="#FFFFFF", alpha=0.28, zorder=0)
        _draw_schematic_world_land(ax)
        for lon in np.arange(-180, 181, 30):
            lw = 0.8 if lon in (-180, 0, 180) else 0.45
            ax.axvline(lon, color="#C9D2DA", lw=lw, ls=":", zorder=2)
        for lat in np.arange(-60, 91, 30):
            lw = 0.8 if lat == 0 else 0.45
            ax.axhline(lat, color="#C9D2DA", lw=lw, ls=":", zorder=2)
    else:
        basemap.drawparallels(
            np.arange(-60, 91, 30),
            labels=[1, 0, 0, 0],
            linewidth=0.35,
            color="#FFFFFF",
            dashes=[2, 2],
            fontsize=8,
            alpha=0.70,
        )
        basemap.drawmeridians(
            np.arange(-180, 181, 60),
            labels=[0, 0, 0, 1],
            linewidth=0.35,
            color="#FFFFFF",
            dashes=[2, 2],
            fontsize=8,
            alpha=0.70,
        )

    scatter = ax.scatter(
        x_values,
        y_values,
        c=plot_df["_arc_affinity"],
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        s=22,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
    )

    if high_arc.any():
        high_df = plot_df.loc[high_arc]
        high_x = x_values[high_arc]
        high_y = y_values[high_arc]
        ax.scatter(
            high_x,
            high_y,
            s=40,
            facecolors="none",
            edgecolors="#1F1F1F",
            linewidths=0.75,
            zorder=5,
            label=f"High-conf. arc-related ≥ {high_threshold:.1f}",
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    if basemap is None:
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 45))
    ax.set_xlabel("Longitude", labelpad=18)
    ax.set_ylabel("Latitude", labelpad=8)
    ax.set_title(
        "Spatial distribution of arc-related affinity in Archean basalts",
        loc="left", pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.018, fraction=0.035)
    cbar.set_label("Arc-related affinity")
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    if high_arc.any():
        ax.legend(loc="lower left", framealpha=0.92, fontsize=8)

    note = (
        f"n = {len(plot_df)}; outlined = {int(high_arc.sum())} high-confidence arc-related samples"
    )
    ax.text(
        0.995, 0.015, note,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color=COLOR_TEXT_MUTED,
    )

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：9 类预测堆积柱状图（Figure 10）
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_stack(df: pd.DataFrame, class_names: list[str], output_path: Path) -> None:
    """
    绘制各年龄段 9 类预测结果的堆积柱状图（v9 §6.2 Figure 10）。
    9 类配色与正文 Figure 1 全球分布图保持一致。X 轴反向，老年代在左。
    """
    counts = (
        df.dropna(subset=["age_bin"])
        .groupby(["age_bin_mid_ma", "pred_class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for name in class_names:
        if name not in counts:
            counts[name] = 0
    counts = counts[class_names].sort_index(ascending=False)
    ratios = counts.div(counts.sum(axis=1), axis=0)

    x       = ratios.index.to_numpy(dtype=float) / 1000.0   # Ma → Ga
    n_bins  = len(x)
    # 柱宽调窄：bin 间距 0.2 Ga，柱宽 ~40% 间距，视觉细瘦
    width   = 0.08 if n_bins > 5 else 0.06

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bottom  = np.zeros(n_bins)

    for name in class_names:
        vals  = ratios[name].to_numpy(dtype=float)
        color = CLASS_COLORS.get(name, "#999999")
        ax.bar(x, vals, bottom=bottom, width=width, color=color,
               label=CLASS_ABBREVS.get(name, name), align="center",
               edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xlabel("Age (Ga)")
    ax.set_ylabel("Predicted class proportion")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # 固定 X 范围以容纳完整 Archean 色块；老在左、年轻在右
    ax.set_xlim(4.00, 2.50)
    ax.set_xticks(np.arange(2.6, 4.01, 0.2))

    # 标题下方为 Archean 色带预留空间（pad 增大）
    ax.set_title("Predicted tectonic affinity through geological time",
                 loc="left", pad=28)
    _add_archean_period_bands(
        ax, band_height=0.055, band_pad=0.010, label_fontsize=9.5,
    )

    # 图例：移至图下方，避免与上方 Archean 色带冲突
    handles = [mpatches.Patch(
        color=CLASS_COLORS.get(n, "#999999"),
        label=CLASS_ABBREVS.get(n, n),
    )
               for n in class_names]
    ax.legend(
        handles=handles,
        ncol=len(class_names),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=8.5,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.5,
        columnspacing=2.5,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：构造环境年龄分布 KDE / ridgeline 山脊图
# ──────────────────────────────────────────────────────────────────────────────

# 旧版 100% 堆叠面积图备份（按需求整体注释，默认改用 KDE 山脊图）。
# def plot_class_streamgraph(df: pd.DataFrame, class_names: list[str],
#                            output_path: Path) -> None:
#     """
#     百分比堆积面积图 — 9 类预测构成比例随年龄演化。
#
#     每个年龄段先按本段样品总数归一到 100%，避免样品量不均衡的时段主导视觉效果。
#     X 轴反向（老年代在左）。
#     """
#     counts = (
#         df.dropna(subset=["age_bin"])
#         .groupby(["age_bin_mid_ma", "pred_class_name"], observed=True)
#         .size()
#         .unstack(fill_value=0)
#     )
#     for name in class_names:
#         if name not in counts:
#             counts[name] = 0
#     counts = counts[class_names].sort_index(ascending=True)
#
#     # 每个年龄段内部归一化，画百分比而不是绝对样品数。
#     totals = counts.sum(axis=1).replace(0, np.nan)
#     ratios = counts.div(totals, axis=0).fillna(0.0) * 100.0
#
#     x = ratios.index.to_numpy(dtype=float) / 1000.0
#     y_arrays = [ratios[c].to_numpy(dtype=float) for c in class_names]
#     colors = [CLASS_COLORS.get(c, "#999999") for c in class_names]
#
#     fig, ax = plt.subplots(figsize=(9.5, 5.4))
#     ax.stackplot(
#         x, *y_arrays, labels=[CLASS_ABBREVS.get(c, c) for c in class_names], colors=colors,
#         edgecolor="white", linewidth=0.4, alpha=0.92,
#     )
#
#     ax.set_xlabel("Age (Ga)")
#     ax.set_ylabel("Predicted class proportion (%)")
#     ax.set_ylim(0, 100)
#     ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
#     ax.set_xlim(4.00, 2.48)
#     ax.set_xticks(np.arange(2.6, 4.01, 0.2))
#     ax.set_title("Predicted tectonic affinity through geological time (100% area)",
#                  loc="left", pad=28)
#     _add_archean_period_bands(
#         ax, band_height=0.055, band_pad=0.010, label_fontsize=9.5,
#     )
#
#     handles = [mpatches.Patch(color=c, label=CLASS_ABBREVS.get(n, n))
#                for c, n in zip(colors, class_names)]
#     ax.legend(
#         handles=handles,
#         ncol=len(class_names),
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.12),
#         fontsize=8.5,
#         frameon=False,
#         handlelength=1.25,
#         handletextpad=0.5,
#         columnspacing=2.5,
#         borderaxespad=0.0,
#     )
#
#     fig.tight_layout()
#     _save_figure(fig, output_path)
#     plt.close(fig)


RIDGELINE_CLASS_ABBREVS = ["BAB", "CA", "CF", "CR", "IOA", "IA", "OI", "OP", "MOR"]
RIDGELINE_ARCHEAN_X_MIN = 2.48
RIDGELINE_ARCHEAN_X_MAX = 4.00


def _smooth_ridge_curve(values: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """中文注释：用简单高斯卷积平滑分箱比例曲线，避免额外依赖 scipy。"""
    if len(values) < 3 or np.allclose(values, 0.0):
        return values

    radius = max(1, int(sigma * 3.0))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_class_kde_ridgeline(
    df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    *,
    x_axis_mode: str = "archean_window",
) -> None:
    """
    绘制构造环境年龄分布 KDE / ridgeline 山脊图。

    中文注释：沿用旧 100% 堆叠面积图的数据口径，先按年龄段统计各类预测比例，
    再对“年龄-比例”序列插值和平滑，得到每个构造环境的一条山脊曲线。
    """
    counts = (
        df.dropna(subset=["age_bin"])
        .groupby(["age_bin_mid_ma", "pred_class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    if counts.empty:
        print("      [跳过] KDE 山脊图：没有有效年龄分箱样品。")
        return

    for name in class_names:
        if name not in counts:
            counts[name] = 0

    class_by_abbrev = {CLASS_ABBREVS.get(name, name): name for name in class_names}
    ordered_classes = [
        class_by_abbrev[abbrev]
        for abbrev in RIDGELINE_CLASS_ABBREVS
        if abbrev in class_by_abbrev
    ]
    counts = counts[ordered_classes].sort_index(ascending=True)

    # 中文注释：每个年龄段内部归一化，曲线高度表示该时代相对预测丰度。
    totals = counts.sum(axis=1).replace(0, np.nan)
    ratios = counts.div(totals, axis=0).fillna(0.0)

    age_ga = ratios.index.to_numpy(dtype=float) / 1000.0
    curve_age_min = max(2.5, float(np.nanmin(age_ga)))
    curve_age_max = min(RIDGELINE_ARCHEAN_X_MAX, float(np.nanmax(age_ga)))

    # 中文注释：保留三种输出范围：完整太古代并两端归零、太古代窗口内仅画数据段、仅当前数据年龄范围。
    if x_axis_mode == "full_archean_zero_edge":
        age_grid = np.linspace(RIDGELINE_ARCHEAN_X_MIN, RIDGELINE_ARCHEAN_X_MAX, 520)
        x_min = RIDGELINE_ARCHEAN_X_MIN
        x_max = RIDGELINE_ARCHEAN_X_MAX
    elif x_axis_mode == "data_age_range":
        age_grid = np.linspace(curve_age_min, curve_age_max, 520)
        x_min = curve_age_min
        x_max = curve_age_max
    else:
        age_grid = np.linspace(curve_age_min, curve_age_max, 520)
        x_min = RIDGELINE_ARCHEAN_X_MIN
        x_max = RIDGELINE_ARCHEAN_X_MAX

    profiles: dict[str, np.ndarray] = {}
    for cls in ordered_classes:
        raw_profile = np.interp(
            age_grid,
            age_ga,
            ratios[cls].to_numpy(dtype=float),
            left=0.0,
            right=0.0,
        )
        profiles[cls] = _smooth_ridge_curve(raw_profile, sigma=12.0)

    max_profile = max(float(profile.max()) for profile in profiles.values())
    height_scale = 0.78 / max_profile if max_profile > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    # 中文注释：主图改为白底，仅保留浅灰网格，避免山脊图整体显得灰蒙蒙。
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    n_cls = len(ordered_classes)
    for row_idx, cls in enumerate(ordered_classes):
        baseline = n_cls - 1 - row_idx
        color = CLASS_COLORS.get(cls, "#999999")
        ridge = profiles[cls] * height_scale

        # 中文注释：先画水平基线，再填充半透明山脊，形成地质文献常见分布图风格。
        ax.hlines(baseline, x_min, x_max, color="#4A4A4A", linewidth=0.65, zorder=1)
        ax.fill_between(
            age_grid,
            baseline,
            baseline + ridge,
            color=color,
            alpha=0.50,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            age_grid,
            baseline + ridge,
            color=color,
            linewidth=1.25,
            zorder=3,
        )

    ax.set_xlim(x_max, x_min)
    ax.set_ylim(-0.20, n_cls - 0.05)
    if x_axis_mode == "data_age_range":
        tick_start = np.ceil(x_min * 10.0) / 10.0
        tick_end = np.floor(x_max * 10.0) / 10.0
        ax.set_xticks(np.arange(tick_start, tick_end + 0.001, 0.2))
    else:
        ax.set_xticks(np.arange(2.6, 4.01, 0.2))
    ax.set_xlabel("Age (Ga)", fontsize=10, labelpad=2)
    ax.set_ylabel("Tectonic setting", fontsize=10, labelpad=0)
    # 中文注释：山脊曲线整体位于基线之上，标签若与基线对齐会显得偏低，
    # 这里把刻度标签整体上移 0.18（数据单位），使其与曲线视觉重心更对齐。
    ax.set_yticks([n_cls - 1 - i + 0.2 for i in range(n_cls)])
    ax.set_yticklabels([CLASS_ABBREVS.get(cls, cls) for cls in ordered_classes],
                       fontsize=10)
    ax.tick_params(axis="y", length=0, pad=6)

    ax.grid(False)
    ax.grid(axis="x", linestyle=(0, (4, 4)), linewidth=0.65,
            color="#C8C8C8", alpha=0.80)

    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)

    # 中文注释：保留旧堆叠图顶部的太古代分期色带，作为年代解释参照。
    _add_archean_period_bands(
        ax,
        band_height=0.060,
        band_pad=0.012,
        label_fontsize=9.0,
    )

    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.13, top=0.88)
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：气泡矩阵（bubble matrix）— 类别 × age bin 稀疏概览
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_bubble_matrix(df: pd.DataFrame, class_names: list[str],
                             output_path: Path) -> None:
    """
    气泡矩阵：行 = 9 类构造背景，列 = age bin（按 Ga 排列）。
    每格圆面积 ∝ 该 bin 中该类样品数；空格代表零样品。

    优势：
      · 比堆积柱更适合展示**稀疏类别**（如 IOA、IA 在某些 bin 仅 1-2 个样品）；
      · 同时编码"绝对数量"和"类别 × 时段"两个维度，比 heatmap 更直观；
      · 视觉上轻盈，适合 SI 或对比附图。
    """
    counts = (
        df.dropna(subset=["age_bin"])
        .groupby(["age_bin_mid_ma", "pred_class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for name in class_names:
        if name not in counts:
            counts[name] = 0
    counts = counts[class_names].sort_index(ascending=False)  # 老在前

    age_ga = counts.index.to_numpy(dtype=float) / 1000.0
    n_age  = len(age_ga)
    n_cls  = len(class_names)

    # 圆面积按全局最大计数缩放，保证最大圆在视觉上 ~150–200 pt²
    max_count = max(int(counts.values.max()), 1)
    size_scale = 220.0 / max_count

    fig, ax = plt.subplots(figsize=(10.0, 0.55 * n_cls + 1.6))
    for j, cls in enumerate(class_names):
        for i, ga in enumerate(age_ga):
            cnt = int(counts.iloc[i, j])
            if cnt > 0:
                ax.scatter(
                    ga, j, s=cnt * size_scale,
                    color=CLASS_COLORS.get(cls, "#999999"),
                    alpha=0.85, edgecolors="white", linewidths=0.7,
                    zorder=3,
                )
                # 中等以上数量的圆中心标注数字
                if cnt >= max(3, max_count * 0.15):
                    ax.text(ga, j, str(cnt),
                            ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold",
                            zorder=4)

    # ── 坐标轴 ──
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels([f"{CLASS_ABBREVS.get(c, c)} — {c}" for c in class_names],
                       fontsize=9)
    ax.invert_yaxis()  # 与列表顺序一致（CA 在最上）
    ax.set_xlabel("Age (Ga)")
    ax.invert_xaxis()
    ax.set_xticks(age_ga)
    ax.set_xticklabels([f"{a:.1f}" for a in age_ga], fontsize=9)
    ax.set_ylim(n_cls - 0.5, -0.5)

    # 浅淡的网格便于对位
    ax.grid(True, axis="both", linestyle=":", linewidth=0.5,
            color="0.7", alpha=0.5, zorder=0)

    # ── 大小图例（放右侧外，避免压主图） ──
    legend_sizes = [1, max(2, max_count // 4), max(5, max_count // 2), max_count]
    legend_handles = [
        plt.scatter([], [], s=s * size_scale, color="0.5", alpha=0.7,
                    edgecolors="white", linewidths=0.7, label=str(s))
        for s in legend_sizes
    ]
    ax.legend(handles=legend_handles, title="Sample count",
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, title_fontsize=9, frameon=True, edgecolor="0.7",
              labelspacing=1.5, borderpad=1.0)

    ax.set_title("Predicted tectonic affinity — class × age bubble matrix",
                 loc="left", pad=6)

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：传统弧型指标对照（Figure 12）
# ──────────────────────────────────────────────────────────────────────────────

def plot_indicator_comparison(indicator_summary: pd.DataFrame, output_path: Path) -> None:
    """
    绘制传统地化指标（Ba/Th、Th/Nb、Ba/La）随年龄变化的对比图（v9 §6.2 Figure 12）。

    每个指标一个子图，对比"所有样品"和"高置信弧相关样品"两组均值。
    子图编号 (a)/(b)/(c) 内嵌左上角粗体。
    """
    SUBSET_STYLES = {
        "all": {
            "color": COLOR_PRIMARY,   "marker": "o",
            "label": "All samples",
            "lw": 1.6, "ms": 5.0, "ls": "-",
        },
        "high_9class_arc": {
            "color": COLOR_HIGHLIGHT, "marker": "s",
            "label": "High-conf. arc-related",
            "lw": 1.6, "ms": 5.0, "ls": "--",
        },
    }
    METRICS = [
        ("mean_Ba_Th", "Ba/Th"),
        ("mean_Th_Nb", "Th/Nb"),
        ("mean_Ba_La", "Ba/La"),
    ]
    PANEL_LABELS = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.4), sharex=True)
    for ax, (metric, ylabel), panel in zip(axes, METRICS, PANEL_LABELS):
        for subset, sty in SUBSET_STYLES.items():
            sub = indicator_summary[indicator_summary["subset"].eq(subset)]
            ax.plot(
                sub["age_mid_ga"], sub[metric],
                color=sty["color"], marker=sty["marker"],
                markersize=sty["ms"], linewidth=sty["lw"], ls=sty["ls"],
                markerfacecolor="white", markeredgewidth=1.2,
                label=sty["label"],
            )
        ax.set_ylabel(f"{ylabel} ratio")
        # 子图编号：内嵌左上角，粗体（v9 §6.2 多子图编号规范）
        _add_panel_label(ax, panel)

    axes[-1].set_xlabel("Age (Ga)")
    # 固定共享 X 轴范围以覆盖完整 Archean
    axes[0].set_xlim(4.00, 2.48)
    axes[0].set_xticks(np.arange(2.6, 4.01, 0.2))
    axes[0].legend(loc="upper right", framealpha=0.92)

    # 仅在顶部子图上方添加 Archean 色带（共享 X 轴下，色带横向覆盖即可）
    _add_archean_period_bands(
        axes[0], band_height=0.07, band_pad=0.020, label_fontsize=9.5,
    )

    # 用 suptitle 时为色带预留空间（y 上调）
    # fig.suptitle("Traditional geochemical arc indicators",
    #              fontsize=11, fontweight="bold", y=1.015)
    fig.tight_layout(h_pad=0.6)
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化：主拼图（composite）—— (a) 山脊图 + (b) 弧相关比例 + (c) 传统地化指标
#   说明：这是一张“组合主图”，复用各原图的数据口径与画法，但全部重绘在一张
#   figure 的多个子图上。为遵守“原画图代码不动”的要求，下面所有逻辑都内联在
#   本函数内，不调用、也不修改 plot_class_kde_ridgeline / plot_arc_ratio /
#   plot_indicator_comparison 等既有函数。
# ──────────────────────────────────────────────────────────────────────────────

# 主拼图专用太古代分期配色（蓝/粉/橙/绿），仅供 plot_main_composite_figure 使用，
# 不影响其它图沿用的 ARCHEAN_PERIODS 配色。
MAIN_PERIOD_COLORS = {
    "Eoarchean":    "#C8D9EC",  # 淡蓝
    "Paleoarchean": "#F0CADA",  # 淡粉
    "Mesoarchean":  "#FAE5C2",  # 淡橙
    "Neoarchean":   "#CDE8C7",  # 淡绿
}

# 转折点标记：每项 (年龄 Ga, 分期配色键, 缓冲半宽 Ga)。
# 以“半透明缓冲带 + 居中虚线”的形式标注，位置可在此随意微调。
MAIN_TRANSITION_MARKERS = [
    (3.80, "Eoarchean",    0.02),
    (3.50, "Paleoarchean", 0.02),
    (3.10, "Mesoarchean",  0.02),
    (2.60, "Neoarchean",   0.02),
]

# 主拼图统一字号：刻度加粗、图例适当放大。
_MAIN_LEGEND_FS = 10.0
_MAIN_TICK_FS   = 9.5


def _main_bold_ticks(ax: plt.Axes) -> None:
    """把 x/y 轴刻度标注统一加粗、略放大。"""
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_fontweight("bold")
        lbl.set_fontsize(_MAIN_TICK_FS)


def _main_period_bands(ax: plt.Axes, *, band_height: float, band_pad: float,
                       label_fontsize: float, show_labels: bool = True) -> None:
    """在 ax 顶部绘制太古代分期色带（使用 MAIN_PERIOD_COLORS 新配色）。"""
    trans = ax.get_xaxis_transform()
    y0 = 1.0 + band_pad
    y1 = y0 + band_height
    for period in ARCHEAN_PERIODS:
        x_lo = min(period["start_ga"], period["end_ga"])
        x_hi = max(period["start_ga"], period["end_ga"])
        color = MAIN_PERIOD_COLORS.get(period["name"], period["color"])
        ax.add_patch(mpatches.Rectangle(
            (x_lo, y0), x_hi - x_lo, band_height,
            transform=trans, facecolor=color, edgecolor="white",
            linewidth=0.6, clip_on=False, zorder=5,
        ))
        if show_labels:
            ax.text(
                (x_lo + x_hi) / 2.0, (y0 + y1) / 2.0, period["name"],
                transform=trans, ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold",
                color="#2A2A2A", clip_on=False, zorder=6,
            )


def _main_transition_markers(ax: plt.Axes) -> None:
    """绘制“缓冲带 + 虚线”形式的转折点标记（置于数据曲线下方）。"""
    for age, key, half in MAIN_TRANSITION_MARKERS:
        color = MAIN_PERIOD_COLORS.get(key, "#999999")
        ax.axvspan(age - half, age + half, color=color, alpha=0.32,
                   linewidth=0.0, zorder=0)
        ax.axvline(age, color="#5A5A5A", linestyle=(0, (5, 4)),
                   linewidth=0.9, zorder=1)


def plot_main_composite_figure(
    df: pd.DataFrame,
    class_names: list[str],
    probs_mean: np.ndarray,
    age_summary: pd.DataFrame,
    liu_age_summary: pd.DataFrame,
    indicator_summary: pd.DataFrame,
    output_path: Path,
    *,
    high_threshold: float = MAIN_ARC_RATIO_THRESHOLD,
    high_std: float = HIGH_STD,
) -> None:
    """
    组合主图：左列上 (a) 9 类构造环境 KDE 山脊图、左列下 (b) 弧相关 affinity 曲线，
    右列 (c) 传统地化指标（Ba/Th、Th/Nb、Ba/La）三联子图。

    四个太古代分期顶带改用蓝/粉/橙/绿新配色；各面板叠加“缓冲带 + 虚线”转折点标记；
    x/y 刻度统一加粗、图例字号放大。
    """
    x_min, x_max = RIDGELINE_ARCHEAN_X_MIN, RIDGELINE_ARCHEAN_X_MAX  # 2.48 / 4.00
    x_ticks = np.arange(2.6, 4.01, 0.2)

    fig = plt.figure(figsize=(16.0, 9.2))
    fig.patch.set_facecolor("#FFFFFF")
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.06, 0.94], wspace=0.18,
        left=0.055, right=0.985, top=0.90, bottom=0.075,
    )
    gs_left  = outer[0, 0].subgridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.31)
    gs_right = outer[0, 1].subgridspec(3, 1, hspace=0.12)
    ax_a = fig.add_subplot(gs_left[0])
    ax_b = fig.add_subplot(gs_left[1])
    ax_c = [fig.add_subplot(gs_right[i]) for i in range(3)]

    # ══ 面板 (a)：9 类构造环境 KDE 山脊图（复用 plot_class_kde_ridgeline 口径） ══
    counts = (
        df.dropna(subset=["age_bin"])
        .groupby(["age_bin_mid_ma", "pred_class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for name in class_names:
        if name not in counts:
            counts[name] = 0
    class_by_abbrev = {CLASS_ABBREVS.get(n, n): n for n in class_names}
    ordered_classes = [
        class_by_abbrev[ab] for ab in RIDGELINE_CLASS_ABBREVS if ab in class_by_abbrev
    ]
    n_cls = len(ordered_classes)
    ax_a.set_facecolor("#FFFFFF")
    if not counts.empty and n_cls > 0:
        counts = counts[ordered_classes].sort_index(ascending=True)
        totals = counts.sum(axis=1).replace(0, np.nan)
        ratios = counts.div(totals, axis=0).fillna(0.0)
        age_ga = ratios.index.to_numpy(dtype=float) / 1000.0
        age_grid = np.linspace(x_min, x_max, 520)
        profiles = {
            cls: _smooth_ridge_curve(
                np.interp(age_grid, age_ga, ratios[cls].to_numpy(dtype=float),
                          left=0.0, right=0.0),
                sigma=12.0,
            )
            for cls in ordered_classes
        }
        max_profile = max((float(p.max()) for p in profiles.values()), default=0.0)
        height_scale = 0.78 / max_profile if max_profile > 0 else 1.0

        _main_transition_markers(ax_a)
        for row_idx, cls in enumerate(ordered_classes):
            baseline = n_cls - 1 - row_idx
            color = CLASS_COLORS.get(cls, "#999999")
            ridge = profiles[cls] * height_scale
            ax_a.hlines(baseline, x_min, x_max, color="#4A4A4A", linewidth=0.65, zorder=2)
            ax_a.fill_between(age_grid, baseline, baseline + ridge,
                              color=color, alpha=0.50, linewidth=0.0, zorder=3)
            ax_a.plot(age_grid, baseline + ridge, color=color, linewidth=1.25, zorder=4)

        ax_a.set_ylim(-0.20, n_cls - 0.05)
        ax_a.set_yticks([n_cls - 1 - i + 0.2 for i in range(n_cls)])
        ax_a.set_yticklabels([CLASS_ABBREVS.get(c, c) for c in ordered_classes])
    ax_a.set_xlim(x_max, x_min)
    ax_a.set_xticks(x_ticks)
    ax_a.set_xlabel("Age (Ga)", fontsize=11)
    ax_a.set_ylabel("Tectonic setting", fontsize=11)
    ax_a.tick_params(axis="y", length=0, pad=6)
    ax_a.grid(False)
    ax_a.grid(axis="x", linestyle=(0, (4, 4)), linewidth=0.65, color="#C8C8C8", alpha=0.80)
    # period 色带统一到 draw() 之后按物理高度换算绘制（见下方），此处不再单独画。
    _main_bold_ticks(ax_a)

    # ══ 面板 (b)：弧相关 affinity（复用 plot_arc_ratio 口径） ══════════════════
    plot_age_ga = age_summary["age_mid_ga"].to_numpy(dtype=float)
    vis = plot_age_ga >= 2.5
    plot_summary = age_summary.loc[vis].reset_index(drop=True)
    x = plot_age_ga[vis]
    thr_summary = _compute_arc_ratio_by_threshold(df, probs_mean, high_threshold, high_std)
    thr_vis = thr_summary["age_mid_ga"] >= 2.5
    x_high = thr_summary.loc[thr_vis, "age_mid_ga"].to_numpy(dtype=float)
    y_high = thr_summary.loc[thr_vis, "ratio_arc"].to_numpy(dtype=float)
    liu_vis = liu_age_summary["age_mid_ga"] >= 2.5
    x_liu = liu_age_summary.loc[liu_vis, "age_mid_ga"].to_numpy(dtype=float)
    y_liu = liu_age_summary.loc[liu_vis, "ratio_liu_arc"].to_numpy(dtype=float)

    _main_transition_markers(ax_b)
    ax_b.text(
        (PT_TRANSITION_GA[0] + PT_TRANSITION_GA[1]) / 2.0, 0.96,
        "Plate tectonic transition\n(Brown et al., 2020)",
        ha="center", va="top", fontsize=8.5, color=COLOR_TEXT_MUTED,
        style="italic", transform=ax_b.get_xaxis_transform(), zorder=2,
    )
    b_series = [
        (x, plot_summary["mean_arc_probability"].to_numpy(dtype=float),
         "Mean arc probability", _ARC_LINE_STYLES[0]),
        (x_high, y_high, f"Samples with P$_{{arc}}$ ≥ {high_threshold:.1f}", _ARC_LINE_STYLES[1]),
        (x_liu, y_liu,
         "Liu et al., 2024 ( ≥ 0.5)", _ARC_LINE_STYLES[2]),
    ]
    for x_v, y_v, label, sty in b_series:
        ax_b.plot(x_v, y_v, color=sty["color"], marker=sty["marker"],
                  markersize=sty["ms"], linewidth=sty["lw"], linestyle=sty["ls"],
                  markerfacecolor="white", markeredgewidth=1.3, zorder=3, label=label)
    ax_b.set_xlabel("Age (Ga)", fontsize=11)
    ax_b.set_ylabel("Arc-related affinity", fontsize=11)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_b.set_xlim(x_max, x_min)
    ax_b.set_xticks(x_ticks)
    ax_b.legend(loc="upper left", framealpha=0.92, fontsize=_MAIN_LEGEND_FS)
    _main_bold_ticks(ax_b)

    # ══ 面板 (c)：传统地化指标三联子图（复用 plot_indicator_comparison 口径） ══
    # 中文注释：弱化 All samples（细线、低透明度蓝色、无 marker，作背景参考线），
    # 突出 High-confidence（粗实线 + 方块 marker，置于上层）。
    SUBSET_STYLES = {
        "all": {"color": COLOR_PRIMARY, "marker": "None", "label": "All samples",
                "lw": 1.0, "ms": 0.0, "ls": "-", "alpha": 0.35, "zorder": 2},
        "high_9class_arc": {"color": COLOR_HIGHLIGHT, "marker": "s",
                            "label": "High-conf. arc-related", "lw": 1.9, "ms": 5.5,
                            "ls": "-", "alpha": 1.0, "zorder": 4},
    }
    METRICS = [("mean_Ba_Th", "Ba/Th"), ("mean_Th_Nb", "Th/Nb"), ("mean_Ba_La", "Ba/La")]
    for idx, (ax, (metric, ylabel)) in enumerate(zip(ax_c, METRICS)):
        _main_transition_markers(ax)
        for subset, sty in SUBSET_STYLES.items():
            sub = indicator_summary[indicator_summary["subset"].eq(subset)]
            ax.plot(sub["age_mid_ga"], sub[metric], color=sty["color"],
                    marker=sty["marker"], markersize=sty["ms"], linewidth=sty["lw"],
                    ls=sty["ls"], markerfacecolor="white", markeredgewidth=1.2,
                    alpha=sty["alpha"], zorder=sty["zorder"], label=sty["label"])
        ax.set_ylabel(f"{ylabel} ratio", fontsize=11)
        ax.set_xlim(x_max, x_min)
        ax.set_xticks(x_ticks)
        if idx < len(ax_c) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Age (Ga)", fontsize=11)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.92, fontsize=_MAIN_LEGEND_FS)
        _main_bold_ticks(ax)

    # ── 顶部分期色带 + 面板编号：统一在 draw() 之后绘制 ──
    fig.canvas.draw()  # 触发布局，使 get_position() 反映最终位置

    # band_height/band_pad 用的是各 axes 的比例坐标，而 a/b/c 子图物理高度不同，
    # 直接传同一个 0.08 会“看起来不一样厚”。这里按各 axes 物理高度反比换算，
    # 把色带高度与上缘留白都统一成占整张图高度的固定比例，使三处看起来一致。
    TARGET_BAND_FIG_FRAC = 0.024   # 色带物理高度（占 figure 高度比例），整体加厚/变薄改这里
    TARGET_PAD_FIG_FRAC  = 0.010   # 色带与子图之间的留白（同上单位）
    for _ax in (ax_a, ax_b, ax_c[0]):
        _h = _ax.get_position().height
        _main_period_bands(
            _ax,
            band_height=TARGET_BAND_FIG_FRAC / _h,
            band_pad=TARGET_PAD_FIG_FRAC / _h,
            label_fontsize=9.5,
        )
    for ax, lab in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c[0], "(c)")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.035, pos.y1 + 0.01, lab,
                 fontsize=14, fontweight="bold", va="bottom", ha="left")
    pos_c = ax_c[0].get_position()
    # fig.text((pos_c.x0 + pos_c.x1) / 2.0, pos_c.y1 + 0.052,
    #          "Traditional geochemical arc indicators",
    #          fontsize=13, fontweight="bold", va="bottom", ha="center")

    _save_figure(fig, output_path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  案例研究预测（6 个克拉通案例）
# ══════════════════════════════════════════════════════════════════════════════

CASE_STUDY_OUTPUT_ROOT = Path(str(ARCHEAN_CASE_DIR))
CASE_RAW_OUTPUT_DIR = CASE_STUDY_OUTPUT_ROOT / "raw"
CASE_PREPROCESSED_OUTPUT_DIR = CASE_STUDY_OUTPUT_ROOT / "preprocessed"
CASE_PREDICTIONS_OUTPUT_DIR = CASE_STUDY_OUTPUT_ROOT / "predictions"
CASE_SUMMARY_CSV_PATH = CASE_PREDICTIONS_OUTPUT_DIR / "case_study_summary.csv"
# 中文注释：左栏六联柱状图、右栏高弧KDE山脊图的组合主图。
CASE_FIG_COMBINED_PATH = CASE_PREDICTIONS_OUTPUT_DIR / "fig_case_studies_bars_ridgeline.png"


def predict_one_case(
    case_config: CaseStudyConfig,
    class_names: list[str],
) -> Optional[pd.DataFrame]:
    """保存案例原始表，严格预处理后再用缺失编码模型执行预测。"""
    # 中文注释：案例读取、路径和44-53 wt%筛选统一由预处理模块负责。
    preprocessed = preprocess_case_study(case_config)
    if preprocessed is None:
        return None
    print(
        f"  严格预处理后 {len(preprocessed)}"
    )

    metadata, normalized = _prepare_archean_features_from_metadata(
        preprocessed,
    )
    if metadata.empty:
        print(f"  [警告] [{case_config.case_label}] 没有有效样品，跳过。")
        return None
    missing_mask = _build_archean_missing_mask(metadata)
    probabilities = _predict_with_missing_mask(
        normalized,
        missing_mask,
        class_names,
        FINAL_MODEL_WEIGHT_PATH,
    )
    pred = add_prediction_columns(
        metadata,
        probabilities,
        np.zeros_like(probabilities),
        class_names,
        high_prob=HIGH_PROB,
        high_std=HIGH_STD,
    )

    pred["case_label"] = case_config.case_label
    pred["approx_age_ga"] = case_config.approx_age_ga

    case_config.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(case_config.predictions_path, index=False, encoding="utf-8-sig")
    return pred


def summarize_case_studies(
    case_results: dict[str, pd.DataFrame],
    class_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """生成 6 案例 × 各类别样品数 + 弧相关比例的汇总 CSV。"""
    rows = []
    for case_label, _, _ in CASE_STUDIES_ORDER:
        if case_label not in case_results:
            continue
        df = case_results[case_label]
        n = len(df)
        arc_n = int(df["pred_class_name"].isin(ARC_RELATED_LABELS).sum())
        high_n = int(df["is_high_confidence"].sum())
        high_arc_n = int((df["is_high_confidence"]
                          & df["pred_class_name"].isin(ARC_RELATED_LABELS)).sum())
        row = {
            "case_label": case_label,
            "n_samples": n,
            "arc_related_count": arc_n,
            "arc_related_pct": arc_n / n if n else np.nan,
            "high_confidence_count": high_n,
            "high_confidence_arc_count": high_arc_n,
        }
        for cls in class_names:
            cnt = int((df["pred_class_name"] == cls).sum())
            row[f"n_{CLASS_ABBREVS.get(cls, cls)}"] = cnt
        rows.append(row)
    summary = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 案例研究图 1：6 子图横向柱状（统一类别行高 + 克制配色）
# ──────────────────────────────────────────────────────────────────────────────

def plot_case_studies_bars(
    case_results: dict[str, pd.DataFrame],
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    6 个案例的子图横向柱状图（3×2 布局）。
    每个子图只显示该案例实际出现的类别，并按数量从大到小排列。
    通过固定 y 轴范围保证不同子图里的横向柱视觉高度一致。
    """
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.5))

    # 仅本图子图标题使用的克拉通展示名（未列出的沿用 CASE_STUDIES_ORDER 中的 case_title）。
    for ax_idx, (case_label, case_title, _) in enumerate(CASE_STUDIES_ORDER):
        ax = axes.flat[ax_idx]
        if case_label not in case_results:
            ax.text(0.5, 0.5, f"{case_title}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="0.5")
            ax.set_axis_off()
            continue

        df = case_results[case_label]
        n = len(df)
        arc_n = int(df["pred_class_name"].isin(ARC_RELATED_LABELS).sum())
        arc_pct = arc_n / n * 100 if n else 0

        counts = df["pred_class_name"].value_counts()
        # 只显示实际出现的类别，并按数量从大到小排列。
        labels_full = [c for c in counts.index if int(counts.get(c, 0)) > 0]
        values      = np.array([int(counts[c]) for c in labels_full])
        labels_abbr = [CLASS_ABBREVS.get(c, c) for c in labels_full]
        colors      = [CASE_BAR_COLORS.get(c, "#7A7A7A") for c in labels_full]

        # 按实际类别数量在固定高度内均匀铺开，并给上下边缘留出呼吸感。
        # 类别少时边缘留白更大，类别多时留白自动缩小。
        if len(labels_full) == 1:
            y_pos = np.array([(len(class_names) - 1) / 2.0])
        else:
            y_span = len(class_names) - 1
            edge_margin = 0.3 + (len(class_names) - len(labels_full)) / (len(class_names) - 1) * 1.45
            y_pos = np.linspace(edge_margin, y_span - edge_margin, len(labels_full))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white",
                       linewidth=0.6, height=0.6)

        # 在每根柱右侧标 count。
        max_v = max(int(values.max()), 1)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + max_v * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(v)), va="center", ha="left",
                    fontsize=12, fontweight="bold", color="0.2")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_abbr, fontsize=12)
        # 稍微拉开 y 轴类别标签和子图绘图区的距离，避免缩写贴近左边框。
        ax.tick_params(axis="y", pad=12)
        ax.invert_yaxis()
        ax.set_ylim(len(class_names) - 0.5, -0.5)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_xlim(0, max_v * 1.20)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_title(
            f"{case_title}   (n = {n},  arc-related = {arc_pct:.0f}%)",
            loc="left", pad=4, fontsize=15,
        )
        ax.grid(False)
        ax.set_axisbelow(True)
        _add_panel_label(ax, f"({chr(ord('a') + ax_idx)})",
                         x=-0.14, y=1.12, fontsize=16)

    # fig.suptitle(
    #     "Predicted tectonic settings of six Archean greenstone-belt cases",
    #     fontsize=15, fontweight="bold", y=1.005,
    # )
    fig.tight_layout()
    # 在 tight_layout 之后再拉大左右两列的横向间距（wspace 越大间距越大）。
    fig.subplots_adjust(wspace=0.25)
    _save_figure(fig, output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 案例研究主流程
# ──────────────────────────────────────────────────────────────────────────────

def run_case_studies(
    class_names: list[str],
) -> dict[str, pd.DataFrame]:
    """
    独立读取六个现成案例数据表，输出三张组合图和一张汇总表。
    返回每个案例的预测 DataFrame 字典。
    """
    print("\n" + "#" * 80)
    print(f"# 案例研究预测（共 {len(CASE_STUDIES_ORDER)} 个克拉通案例）")
    print("#" * 80)

    CASE_RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASE_PREPROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASE_PREDICTIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_results: dict[str, pd.DataFrame] = {}

    for case_config in build_case_study_configs():
        print(f"\n[案例] {case_config.case_title} (~{case_config.approx_age_ga} Ga)")
        pred = predict_one_case(case_config, class_names)
        if pred is None:
            continue
        n = len(pred)
        arc_n = int(pred["pred_class_name"].isin(ARC_RELATED_LABELS).sum())
        print(f"  → 预测样品 {n}, 弧相关 {arc_n} ({arc_n/n*100:.1f}%)")
        case_results[case_config.case_label] = pred

    if not case_results:
        print("⚠ 无案例预测结果产生，跳过案例图绘制。")
        return case_results

    # 汇总 CSV
    summarize_case_studies(case_results, class_names, CASE_SUMMARY_CSV_PATH)
    print(f"\n  汇总 CSV: {CASE_SUMMARY_CSV_PATH}")

    # 中文注释：案例研究统一输出左右双栏主图。
    from archean_case_studies_map_ridgeline import plot_case_studies_bars_ridgeline

    print("  绘制案例双栏主图: 左侧六联柱状图，右侧高弧KDE山脊图")
    plot_case_studies_bars_ridgeline(case_results, CASE_FIG_COMBINED_PATH)

    print(f"\n案例研究完成。输出根目录: {CASE_STUDY_OUTPUT_ROOT}")
    return case_results


# ──────────────────────────────────────────────────────────────────────────────
# Markdown 报告
# ──────────────────────────────────────────────────────────────────────────────

def _format_markdown_cell(value: object) -> str:
    """将单元格值转换为 Markdown 表格可安全显示的文本。"""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("\n", "<br>").replace("|", r"\|")


def dataframe_to_markdown_table(table: pd.DataFrame) -> str:
    """生成 Markdown 表格，避免依赖 pandas.to_markdown 所需的 tabulate。"""
    if table.empty:
        return "_No rows._"

    # 手动拼接 Markdown 表格，避免缺少 tabulate 时报告生成失败。
    headers = [_format_markdown_cell(column) for column in table.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for row in table.itertuples(index=False, name=None):
        cells = [_format_markdown_cell(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    age_summary: pd.DataFrame,
    craton_summary: pd.DataFrame,
    config: RunConfig,
    recommended_thresh: Optional[float] = None,
) -> None:
    """生成 Markdown 格式的分析报告。"""
    n          = len(df)
    high_n     = int(df["is_high_confidence"].sum())
    arc_high_n = int((df["is_high_confidence"] & df["is_arc_related_9class"]).sum())
    
    # Liu 2024 基线统计（只统计有 Liu 预测值的样品）
    liu_valid = df[df["liu_arc_binary"].notna()]
    n_liu_valid = len(liu_valid)
    liu_arc_n = int(liu_valid["liu_arc_binary"].sum()) if n_liu_valid > 0 else 0
    
    mean_missing = float(df["missing_feature_count_36"].mean()) if "missing_feature_count_36" in df else np.nan
    max_missing = int(df["missing_feature_count_36"].max()) if "missing_feature_count_36" in df else 0

    top_cratons = dataframe_to_markdown_table(craton_summary.head(10)[
        ["Craton", "n_all", "n_9class_arc_high", "oldest_9class_arc_high_C_AGE_Ma"]
    ])

    age_summary_table = dataframe_to_markdown_table(age_summary[[
        "age_bin", "n_all", "n_high",
        "ratio_9class_arc_all", "ratio_9class_arc_high", "n_liu_samples", "ratio_liu_arc",
    ]])

    thresh_note = (
        f"- Coverage-Accuracy diagnostic threshold: `{recommended_thresh:.3f}`"
        if recommended_thresh is not None
        else "- Coverage-Accuracy curve: skipped or unavailable in current Python environment"
    )
    threshold_mode = (
        "coverage-accuracy diagnostic threshold"
        if config.use_coverage_recommended_threshold and recommended_thresh is not None
        else "fixed configured HIGH_PROB"
    )

    lines = [
        "# Archean basalt ViT-Transformer Dual-Stream prediction report",
        "",
        f"- Preprocess variant: `{config.preprocess_variant}`",
        f"- Source CSV (Liu et al. 2024 original): `{config.source_s3_csv_path}`",
        f"- Preprocessed sample table: `{config.preprocessed_s3_path}`",
        f"- Preprocessed feature table: `{config.preprocessed_features_path}`",
        f"- Model weights: `{'; '.join(str(p) for p in config.weights)}`",
        f"- Samples entering model: {n}",
        f"- Missing features in source data: mean {mean_missing:.2f}, max {max_missing}",
        f"- High-confidence threshold used: `{config.high_prob:.3f}` ({threshold_mode})",
        f"- High-confidence samples: {high_n} ({high_n / n:.1%})",
        f"- High-confidence arc-related samples: {arc_high_n} ({arc_high_n / n:.1%} of all samples)",
        f"- Liu 2024 baseline: {n_liu_valid} samples with original predictions, {liu_arc_n} arc-related (`Arc_probability3 >= 0.5`, {liu_arc_n / n_liu_valid:.1%})",
        thresh_note,
        "",
        "Arc-related classes: Continental arc, Intra-oceanic arc, Island arc.",
        "",
        "## Age-bin summary",
        "",
        age_summary_table,
        "",
        "## Top cratons by oldest high-confidence arc-related signal",
        "",
        top_cratons,
        "",
    ]
    config.analysis_report_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 主流水线
# ──────────────────────────────────────────────────────────────────────────────

QUANTILE_PARAMS_PATH = Path(str(QUANTILE_PARAMS_JSON))

ARCHEAN_COLUMN_MAPPING = {
    "SIO2": "SIO2(WT%)", "TIO2": "TIO2(WT%)",
    "AL2O3": "AL2O3(WT%)", "FEOT": "FEOT(WT%)",
    "MNO": "MNO(WT%)", "MGO": "MGO(WT%)", "CAO": "CAO(WT%)",
    "NA2O": "NA2O(WT%)", "K2O": "K2O(WT%)", "P2O5": "P2O5(WT%)",
    "V": "V(PPM)", "CR": "CR(PPM)", "CO": "CO(PPM)", "NI": "NI(PPM)",
    "RB": "RB(PPM)", "SR": "SR(PPM)", "Y": "Y(PPM)", "ZR": "ZR(PPM)",
    "NB": "NB(PPM)", "BA": "BA(PPM)", "LA": "LA(PPM)", "CE": "CE(PPM)",
    "PR": "PR(PPM)", "ND": "ND(PPM)", "SM": "SM(PPM)", "EU": "EU(PPM)",
    "GD": "GD(PPM)", "TB": "TB(PPM)", "DY": "DY(PPM)", "HO": "HO(PPM)",
    "ER": "ER(PPM)", "YB": "YB(PPM)", "LU": "LU(PPM)", "HF": "HF(PPM)",
    "TA": "TA(PPM)", "TH": "TH(PPM)",
}
ARCHEAN_MAJOR_COLUMNS = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)",
    "P2O5(WT%)", "K2O(WT%)", "CAO(WT%)", "TIO2(WT%)",
    "MNO(WT%)", "FEOT(WT%)",
]


def _extract_archean_feature_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """兼容扩展总表的短列名和案例原始表的标准长列名。"""
    features = pd.DataFrame(index=metadata.index)
    for short_name, long_name in ARCHEAN_COLUMN_MAPPING.items():
        if short_name in metadata.columns:
            source_column = short_name
        elif long_name in metadata.columns:
            source_column = long_name
        else:
            features[long_name] = np.nan
            continue
        features[long_name] = pd.to_numeric(
            metadata[source_column],
            errors="coerce",
        )
    return features[COLUMNS_TO_EXTRACT].where(
        features[COLUMNS_TO_EXTRACT] > 0
    )


def _prepare_archean_features(
    archean_path: Path,
    max_missing_features: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将太古代原始数据转换为1-255编码，缺失单元固定保留为0。"""
    metadata = read_csv_fallback(archean_path)
    return _prepare_archean_features_from_metadata(
        metadata,
        max_missing_features=max_missing_features,
    )


def _prepare_archean_features_from_metadata(
    metadata: pd.DataFrame,
    max_missing_features: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将内存中的太古代数据转换为1-255编码，缺失单元固定保留为0。"""
    metadata = metadata.copy()
    features = _extract_archean_feature_table(metadata)

    missing_count = features.isna().sum(axis=1)
    if max_missing_features is None:
        # 中文注释：正文总池只剔除36个特征全部缺失的样品。
        valid_rows = missing_count < len(COLUMNS_TO_EXTRACT)
    else:
        # 中文注释：案例研究最多允许缺失18个特征，即至少保留18个有效特征。
        valid_rows = missing_count <= max_missing_features
    dropped_count = int((~valid_rows).sum())
    metadata = metadata.loc[valid_rows].reset_index(drop=True)
    features = features.loc[valid_rows].reset_index(drop=True)
    metadata["missing_feature_count_36"] = (
        missing_count.loc[valid_rows].to_numpy(dtype=np.int16)
    )

    # 中文注释：只用样品当前已有的主量元素做无水标准化，缺失值不参与计算。
    major_total = features[ARCHEAN_MAJOR_COLUMNS].sum(
        axis=1,
        min_count=1,
    )
    features.loc[:, ARCHEAN_MAJOR_COLUMNS] = features[
        ARCHEAN_MAJOR_COLUMNS
    ].div(major_total, axis=0) * 100.0

    with QUANTILE_PARAMS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        quantile_params = json.load(file)

    encoded = pd.DataFrame(index=features.index)
    for column in COLUMNS_TO_EXTRACT:
        values = features[column].to_numpy(dtype=float)
        boundaries = np.asarray(quantile_params[column], dtype=float)
        valid = np.isfinite(values)
        encoded_values = np.zeros(len(values), dtype=np.int16)
        encoded_values[valid] = (
            np.searchsorted(
                boundaries,
                values[valid],
                side="left",
            ) + 1
        ).clip(1, 255)
        encoded[column] = encoded_values

    print(
        f"      [原始数据] 样品={len(encoded)}，"
        f"缺失筛选剔除={dropped_count}，"
        f"缺失编码0单元={int((encoded == 0).sum().sum())}"
    )
    return metadata, encoded


def _build_archean_missing_mask(metadata: pd.DataFrame) -> pd.DataFrame:
    """根据原始扩展太古代数据生成与训练阶段一致的36维缺失编码。"""
    features = _extract_archean_feature_table(metadata)
    missing_mask = features.isna().astype(np.uint8)
    missing_mask.columns = [
        f"missing_mask__{column}" for column in COLUMNS_TO_EXTRACT
    ]
    return missing_mask.reset_index(drop=True)


def _predict_with_missing_mask(
    normalized: pd.DataFrame,
    missing_mask: pd.DataFrame,
    class_names: list[str],
    weight_path: Path,
) -> np.ndarray:
    """加载缺失编码GeoDAN权重并返回九分类概率。"""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("2×2实验需要PyTorch环境")

    from torch.utils.data import DataLoader, TensorDataset

    x_img, x_seq = build_model_inputs(normalized, missing_mask)
    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(x_img),
            torch.FloatTensor(x_seq),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(
        len(class_names),
        use_missing_mask=True,
    ).to(device)
    model = _load_weights(weight_path, model, device)
    model.eval()

    probability_parts = []
    with torch.no_grad():
        for batch_index, (batch_img, batch_seq) in enumerate(loader, start=1):
            logits = model(
                batch_img.to(device),
                batch_seq.to(device),
            )
            probability_parts.append(
                F.softmax(logits, dim=1).cpu().numpy()
            )
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"        推理批次 {batch_index}/{len(loader)}",
                    flush=True,
                )
    return np.concatenate(probability_parts, axis=0)


FINAL_ARCHEAN_RAW_PATH = Path(str(ARCHEAN_POOL_CSV))
FINAL_ARCHEAN_MASK_PATH = Path(str(ARCHEAN_FINAL_MASK_CSV))
FINAL_MODEL_WEIGHT_PATH = Path(str(MAIN_MODEL_WEIGHT))
FINAL_OUTPUT_DIR = Path(str(ARCHEAN_FINAL_DIR))
FINAL_PREDICTION_PATH = Path(str(ARCHEAN_FINAL_PREDICTIONS_CSV))
FINAL_COMPOSITE_PATH = FINAL_OUTPUT_DIR / "fig_main_composite_tectonic.png"
# 中文注释：三文件Zenodo发布不包含Liu原始表和六个案例区文件。
# 缺少这些可选资料时自动切换为仅预测模式，避免主预测流程被附加图件阻断。
PREDICTION_ONLY = (
    os.environ.get("GEODAN_PREDICTION_ONLY", "0") == "1"
    or not SOURCE_S3_CSV_PATH.exists()
)
def run_final_prediction() -> None:
    """运行固定正文方案：CFB=6920、显式缺失编码、SiO2 44-53 wt%。"""
    required_paths = [
        FINAL_ARCHEAN_RAW_PATH,
        QUANTILE_PARAMS_PATH,
        FINAL_MODEL_WEIGHT_PATH,
        TRAIN_PATH,
    ]
    if not PREDICTION_ONLY:
        required_paths.append(SOURCE_S3_CSV_PATH)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "最终方案缺少文件:\n" + "\n".join(str(path) for path in missing_paths)
        )

    print("=" * 80)
    print("最终方案：CFB=6920 + 缺失编码 + 太古代SiO2 44-53 wt%")
    class_names = load_class_names(TRAIN_PATH)
    # 中文注释：正式3012条总池由统一预处理模块读取和筛选。
    metadata = load_final_age_constrained_pool()
    metadata, normalized = _prepare_archean_features_from_metadata(metadata)
    missing_mask = _build_archean_missing_mask(metadata)
    FINAL_ARCHEAN_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing_mask.to_csv(
        FINAL_ARCHEAN_MASK_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    probabilities = _predict_with_missing_mask(
        normalized,
        missing_mask,
        class_names,
        FINAL_MODEL_WEIGHT_PATH,
    )
    prediction = add_prediction_columns(
        metadata,
        probabilities,
        np.zeros_like(probabilities),
        class_names,
        high_prob=HIGH_PROB,
        high_std=HIGH_STD,
    )
    prediction = add_age_bins(prediction, BIN_SIZE_MYR)

    # 仅预测模式不读取 Liu 原始表，也不运行需要额外案例数据的下游分析。
    if PREDICTION_ONLY:
        FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(
            FINAL_PREDICTION_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"仅预测模式完成，预测结果: {FINAL_PREDICTION_PATH}")
        return

    age_summary = summarize_by_age(prediction, class_names)
    indicator_summary = summarize_indicators(prediction)
    liu_age_summary = summarize_liu_baseline_by_age(
        SOURCE_S3_CSV_PATH,
        BIN_SIZE_MYR,
    )

    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(
        FINAL_PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    plot_main_composite_figure(
        prediction,
        class_names,
        probabilities,
        age_summary,
        liu_age_summary,
        indicator_summary,
        FINAL_COMPOSITE_PATH,
        high_threshold=0.5,
        high_std=HIGH_STD,
    )
    run_case_studies(class_names)

    high_arc_count = int(prediction["Arc_probability3"].ge(0.5).sum())
    cfb_count = int(
        prediction["pred_class_name"].eq(
            "CONTINENTAL FLOOD BASALT"
        ).sum()
    )
    print(
        f"完成：样品={len(prediction)}，高弧={high_arc_count}，"
        f"CFB={cfb_count}"
    )
    print(f"预测结果: {FINAL_PREDICTION_PATH}")
    print(f"正文主图: {FINAL_COMPOSITE_PATH}")
    print(f"案例研究: {CASE_STUDY_OUTPUT_ROOT}")


# ──────────────────────────────────────────────────────────────────────────────
# 入口：正文流程固定为CFB=6920、缺失编码、太古代原始数据
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_final_prediction()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
