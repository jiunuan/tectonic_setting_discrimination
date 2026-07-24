"""
ViT-Transformer 双流玄武岩构造环境分类模型 —— 消融实验（v4）
================================================================
v4 重构说明（相对 v3 多种子版）：
  ※ 主模型架构升级 ※
    Full Model 由 CNN+ViT+Seq Transformer 三件套
    简化为 ViT + Seq Transformer 双流（去掉 CNN，与 EMSPN 形成对称升级叙事）
        矩阵分支: 6×6 亲缘矩阵 → Patch Embed → ViT Encoder
        序列分支: 36 元素相容性序列 → Linear Embed → Transformer Encoder
        融合: 两分支均值池化 → Concat → MLP 分类头

  ※ 模型清单调整 ※
    Full          : ViT-Transformer 双流              （新主模型）
    Abl-1 (新增)  : ViT Only       (仅矩阵分支)
    Abl-2         : Transformer Only (仅序列分支)
    Abl-3         : 双流 w/o Positional Encoding
    Cmp-1         : CNN-BiLSTM    (EMSPN 前作直接对比)
    Cmp-2         : CNN-ViT-Transformer (旧 Full，证明 CNN 冗余)
    Cmp-3         : CNN Only
    ML            : RF / SVM / XGBoost / MLP

  ※ 打印频率 ※
    DL  训练: 每 20 个 epoch 打印一次（首末 epoch 也打印）
    MLP 训练: 每 20 个 epoch 打印一次（首末 epoch 也打印）

  ※ 多种子配置 ※
    SEEDS = [42, 123, 456, 789, 1024]
    输出 ablation_per_seed.csv（明细）+ ablation_summary.csv（mean ± std）
    柱状图带 error bar，训练曲线用最高准确率种子绘制
"""

import collections
import copy
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings('ignore')
# 中文注释：统一从当前项目配置读取数据和模型输出路径，避免保留 CNNtest 本地路径。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import MODELS_DIR, TRAIN_NORM_CSV, TEST_NORM_CSV, MASK_TRAIN_CSV, MASK_TEST_CSV

# ════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════
# SEEDS          = [42, 123, 456, 789, 1024]
SEEDS          = [42, 123]
PRINT_EVERY   = 1   # 中文注释：每10轮打印一次，避免长训练产生过量终端输出。
ML_PRINT_EVERY = 100   # ← v4: ML(MLP) 每 20 个 epoch 打印一次
# 中文注释：SMOTE后关闭类别加权，避免对少数类重复补偿。
USE_CLASS_WEIGHTED_LOSS = False

# SCI 期刊混淆矩阵配色方案（每个模型输出全部版本）
SCI_CMAPS = [
    ('Blues',    'Blues'),      # 经典蓝色序列（最常见）
    # ('YlOrRd',   'YlOrRd'),     # 黄橙红渐变（暖色系）
    # ('Purples',  'Purples'),    # 紫色序列（优雅）
    # ('viridis',  'viridis'),    # 感知均匀（Nature/Science 推荐）
    # ('Oranges',  'Oranges'),    # 橙色序列（暖调清晰）
]

# 中文注释：统一训练输出中的混淆矩阵画法，与 confusion_matrices/data.py 保持一致。
GEODAN_CONFUSION_ORDER = ["CA", "IA", "IOA", "BAB", "MOR", "OP", "OI", "CR", "CF"]
SOFT_GEODAN_CMAP = LinearSegmentedColormap.from_list(
    "soft_geodan_blue_high_end",
    [
        (0.00, "#f8fafc"),
        (0.05, "#edf3f8"),
        (0.10, "#e1edf5"),
        (0.30, "#c8ddeb"),
        (0.60, "#8fb9d4"),
        (0.78, "#4f8fbd"),
        (0.84, "#2f74aa"),
        (0.88, "#1f6099"),
        (0.92, "#124f8c"),
        (0.96, "#0b3f78"),
        (1.00, "#062f62"),
    ],
    N=256,
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def safe_filename(name: str) -> str:
    """将任意字符串转换为 Windows/Linux 均合法的文件名。"""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, '_')
    name = name.replace('\n', '_').replace(' ', '_')
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')


def _reorder_geodan_confusion_matrix(cm_pct, class_names):
    """中文注释：按论文图中的地质类别顺序重排混淆矩阵。"""
    if set(GEODAN_CONFUSION_ORDER).issubset(set(class_names)):
        order_idx = [class_names.index(label) for label in GEODAN_CONFUSION_ORDER]
        return cm_pct[np.ix_(order_idx, order_idx)], GEODAN_CONFUSION_ORDER
    return cm_pct, class_names


def _plot_geodan_confusion_matrix(cm_pct, class_names, save_path):
    """中文注释：绘制与独立混淆矩阵脚本一致的 GeoDAN 风格百分比矩阵。"""
    cm_plot, labels_plot = _reorder_geodan_confusion_matrix(cm_pct, list(class_names))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=600)

    im = ax.imshow(
        cm_plot,
        cmap=SOFT_GEODAN_CMAP,
        vmin=0,
        vmax=100,
        interpolation="nearest",
        aspect="equal",
    )

    # 中文注释：坐标轴标签与独立绘图脚本保持一致。
    ax.set_xticks(np.arange(len(labels_plot)))
    ax.set_yticks(np.arange(len(labels_plot)))
    ax.set_xticklabels(labels_plot, fontsize=10, fontweight="semibold")
    ax.set_yticklabels(labels_plot, fontsize=10, fontweight="semibold")
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

    # 中文注释：细白网格降低模板感，分组线对应 arc/back-arc/oceanic/continental。
    ax.set_xticks(np.arange(-0.5, len(labels_plot), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_plot), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    if len(labels_plot) == len(GEODAN_CONFUSION_ORDER):
        for pos in [3.5, 6.5]:
            ax.axhline(pos, color="#6f7f8f", linewidth=0.35, alpha=0.35)
            ax.axvline(pos, color="#6f7f8f", linewidth=0.35, alpha=0.35)

    # 中文注释：所有格子保留一位小数，深色格子用白字提高可读性。
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            value = cm_plot[i, j]
            text_color = "white" if value >= 55 else "#1f1f1f"
            fontweight = "semibold" if i == j else "normal"
            ax.text(
                j, i, f"{value:.1f}",
                ha="center", va="center",
                fontsize=10,
                color=text_color,
                fontweight=fontweight,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    # 中文注释：colorbar 绑定主图，保证色带高度与矩阵主体一致，并保留稍大的间距。
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.24)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized percentage (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=9, width=0.6)
    cbar.outline.set_linewidth(0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# 矩阵分支输入 V1：6×6 元素周期表顺序（原始方案）
# =============================================================

ORIGINAL_IMAGE_COLUMNS = [
    'NA2O(WT%)', 'MGO(WT%)', 'CR(PPM)',   'AL2O3(WT%)', 'SIO2(WT%)', 'P2O5(WT%)',
    'K2O(WT%)',  'CAO(WT%)', 'TIO2(WT%)', 'V(PPM)',     'MNO(WT%)',  'FEOT(WT%)',
    'RB(PPM)',   'SR(PPM)',  'Y(PPM)',    'NB(PPM)',    'CO(PPM)',   'NI(PPM)',
    'BA(PPM)',   'LA(PPM)',  'CE(PPM)',   'PR(PPM)',    'ND(PPM)',   'ZR(PPM)',
    'SM(PPM)',   'EU(PPM)',  'GD(PPM)',   'TB(PPM)',    'DY(PPM)',   'HO(PPM)',
    'TH(PPM)',   'ER(PPM)',  'YB(PPM)',   'LU(PPM)',    'HF(PPM)',   'TA(PPM)',
]

# 序列分支 V1：电极电势序列
COLUMNS_ELECTRODE_ORDER_V1 = [
    'RB(PPM)',    'K2O(WT%)',  'BA(PPM)',   'SR(PPM)',    'CAO(WT%)',  'NA2O(WT%)',
    'LA(PPM)',    'Y(PPM)',    'MGO(WT%)',  'PR(PPM)',    'CE(PPM)',   'ER(PPM)',
    'HO(PPM)',    'ND(PPM)',   'SM(PPM)',   'DY(PPM)',    'LU(PPM)',   'TB(PPM)',
    'GD(PPM)',    'YB(PPM)',   'EU(PPM)',   'TH(PPM)',    'AL2O3(WT%)','HF(PPM)',
    'ZR(PPM)',    'TIO2(WT%)', 'MNO(WT%)', 'V(PPM)',     'NB(PPM)',   'CR(PPM)',
    'TA(PPM)',    'FEOT(WT%)', 'CO(PPM)',  'NI(PPM)',    'SIO2(WT%)', 'P2O5(WT%)',
]

# =============================================================
# 矩阵分支输入 V2：6×6 地化亲缘矩阵
# =============================================================

IMAGE_GRID_V2 = [
    # 第 0 行：大离子亲石元素（LILE）
    ['RB(PPM)',   'BA(PPM)',    'TH(PPM)',    'SR(PPM)',    'K2O(WT%)',   'NA2O(WT%)' ],
    # 第 1 行：轻稀土（LREE）+ Nb-Ta 夹心
    ['LA(PPM)',   'CE(PPM)',    'NB(PPM)',    'TA(PPM)',    'PR(PPM)',    'ND(PPM)'   ],
    # 第 2 行：中稀土（MREE）+ Zr-Hf 夹心
    ['SM(PPM)',   'EU(PPM)',    'ZR(PPM)',    'HF(PPM)',    'GD(PPM)',    'TB(PPM)'   ],
    # 第 3 行：连续重稀土（HREE）
    ['DY(PPM)',   'HO(PPM)',    'ER(PPM)',    'YB(PPM)',    'LU(PPM)',    'Y(PPM)'    ],
    # 第 4 行：主量氧化物
    ['SIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'MGO(WT%)',   'CAO(WT%)',   'TIO2(WT%)' ],
    # 第 5 行：相容元素 + P
    ['CR(PPM)',   'NI(PPM)',    'CO(PPM)',    'V(PPM)',     'MNO(WT%)',   'P2O5(WT%)' ],
]

IMAGE_COLUMNS_V2 = [col for row in IMAGE_GRID_V2 for col in row]


# =============================================================
# 序列分支输入：正式 v1 按标准电极电势（E°）排列（共 36 列）
# =============================================================

SEQUENCE_COLUMNS_V2 = [
    # ── 高度不相容元素 ──────────────────────────────────
    'RB(PPM)', 'BA(PPM)', 'TH(PPM)', 'K2O(WT%)', 'NA2O(WT%)',
    'NB(PPM)', 'TA(PPM)', 'LA(PPM)', 'CE(PPM)',  
    'PR(PPM)', 'SR(PPM)', 'P2O5(WT%)',
    # ── 中等不相容元素 ──────────────────────────────────
    'ND(PPM)', 'SM(PPM)', 'ZR(PPM)', 'HF(PPM)', 'EU(PPM)',
    'TIO2(WT%)', 'AL2O3(WT%)', 'GD(PPM)', 'TB(PPM)', 'DY(PPM)',
    # ── 相对相容元素 ────────────────────────────────────
    'HO(PPM)',  'Y(PPM)',   'ER(PPM)', 'YB(PPM)', 'LU(PPM)',
    'CAO(WT%)', 'V(PPM)', 'MNO(WT%)',
    # ── 高度相容元素 ────────────────────────────────────
    'FEOT(WT%)', 'MGO(WT%)', 'SIO2(WT%)', 'CR(PPM)', 'NI(PPM)', 'CO(PPM)',
]

COLUMNS_ELECTRODE_ORDER = SEQUENCE_COLUMNS_V2


# =============================================================
# 数据加载
# =============================================================

def reshape_to_image(X_2d):
    return X_2d.reshape(-1, 1, 6, 6).astype(np.float32)


def _load_missing_masks(
    train_mask_path,
    test_mask_path,
    train_row_count,
    test_row_count,
    columns_to_extract,
    seq_columns,
):
    """读取原始缺失mask；SMOTE新增训练行统一记为0。"""
    train_mask_raw = pd.read_csv(train_mask_path, encoding='utf-8-sig')
    test_mask_raw = pd.read_csv(test_mask_path, encoding='utf-8-sig')

    image_mask_columns = [
        f'missing_mask__{column}' for column in columns_to_extract
    ]
    sequence_mask_columns = [
        f'missing_mask__{column}' for column in seq_columns
    ]
    required_columns = sorted(set(image_mask_columns + sequence_mask_columns))
    for dataset_name, mask_data in [
        ('训练集', train_mask_raw),
        ('测试集', test_mask_raw),
    ]:
        missing_columns = [
            column for column in required_columns
            if column not in mask_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f'{dataset_name}缺失mask文件缺少列: {missing_columns}'
            )

    if len(train_mask_raw) > train_row_count:
        raise ValueError('训练集缺失mask行数大于模型训练集行数')
    if len(test_mask_raw) != test_row_count:
        raise ValueError(
            f'测试集缺失mask行数不一致: '
            f'{len(test_mask_raw)} != {test_row_count}'
        )

    synthetic_count = train_row_count - len(train_mask_raw)
    synthetic_mask = pd.DataFrame(
        0,
        index=np.arange(synthetic_count),
        columns=train_mask_raw.columns,
        dtype=np.uint8,
    )
    train_mask = pd.concat(
        [train_mask_raw, synthetic_mask],
        ignore_index=True,
    )

    train_img_mask = reshape_to_image(
        train_mask[image_mask_columns].to_numpy(dtype=np.float32)
    )
    test_img_mask = reshape_to_image(
        test_mask_raw[image_mask_columns].to_numpy(dtype=np.float32)
    )
    train_seq_mask = train_mask[
        sequence_mask_columns
    ].to_numpy(dtype=np.float32)[:, :, np.newaxis]
    test_seq_mask = test_mask_raw[
        sequence_mask_columns
    ].to_numpy(dtype=np.float32)[:, :, np.newaxis]

    print(
        f'  缺失mask: 真实训练行={len(train_mask_raw)}，'
        f'SMOTE新增行={synthetic_count}，测试行={len(test_mask_raw)}'
    )
    print(
        f'  训练mask缺失率='
        f'{train_mask_raw.to_numpy(dtype=float).mean():.2%}，'
        f'测试mask缺失率='
        f'{test_mask_raw.to_numpy(dtype=float).mean():.2%}'
    )
    return train_img_mask, train_seq_mask, test_img_mask, test_seq_mask


def load_presplit_csv(
    train_path,
    test_path,
    columns_to_extract,
    seq_columns=None,
    use_missing_mask=False,
    train_mask_path=None,
    test_mask_path=None,
    cfb_target_count=None,
):
    if seq_columns is None:
        seq_columns = COLUMNS_ELECTRODE_ORDER
    try:
        df_train = pd.read_csv(train_path, encoding='utf-8')
        df_test  = pd.read_csv(test_path,  encoding='utf-8')
    except UnicodeDecodeError:
        df_train = pd.read_csv(train_path, encoding='ISO-8859-1')
        df_test  = pd.read_csv(test_path,  encoding='ISO-8859-1')

    X_train_img_2d = df_train[columns_to_extract].values.astype(np.float32)
    X_test_img_2d  = df_test[columns_to_extract].values.astype(np.float32)

    X_train_seq_2d = df_train[seq_columns].values.astype(np.float32)
    X_test_seq_2d  = df_test[seq_columns].values.astype(np.float32)

    y_train_raw, unique = pd.factorize(df_train['TECTONIC SETTING'])
    label2idx  = {label: idx for idx, label in enumerate(unique)}
    y_test_raw = df_test['TECTONIC SETTING'].map(label2idx).values

    if np.any(pd.isna(y_test_raw)):
        unknown = set(df_test['TECTONIC SETTING'].unique()) - set(label2idx.keys())
        raise ValueError(f'测试集包含未见过的标签: {unknown}')

    y_train = y_train_raw.astype(np.int64)
    y_test  = y_test_raw.astype(np.int64)

    X_train_img = reshape_to_image(X_train_img_2d / 255.0)
    X_test_img  = reshape_to_image(X_test_img_2d  / 255.0)
    X_train_seq = (X_train_seq_2d / 255.0)[:, :, np.newaxis].astype(np.float32)
    X_test_seq  = (X_test_seq_2d  / 255.0)[:, :, np.newaxis].astype(np.float32)

    if use_missing_mask:
        if train_mask_path is None or test_mask_path is None:
            raise ValueError('启用缺失编码时必须提供训练集和测试集mask路径')
        (
            X_train_img_mask,
            X_train_seq_mask,
            X_test_img_mask,
            X_test_seq_mask,
        ) = _load_missing_masks(
            train_mask_path,
            test_mask_path,
            len(df_train),
            len(df_test),
            columns_to_extract,
            seq_columns,
        )
        # 中文注释：矩阵分支使用“数值+mask”两个通道，序列分支每个元素使用两个输入特征。
        X_train_img = np.concatenate(
            [X_train_img, X_train_img_mask], axis=1
        )
        X_test_img = np.concatenate(
            [X_test_img, X_test_img_mask], axis=1
        )
        X_train_seq = np.concatenate(
            [X_train_seq, X_train_seq_mask], axis=2
        )
        X_test_seq = np.concatenate(
            [X_test_seq, X_test_seq_mask], axis=2
        )

    if cfb_target_count is not None:
        cfb_label = 'CONTINENTAL FLOOD BASALT'
        cfb_indices = np.flatnonzero(
            df_train['TECTONIC SETTING'].astype(str).to_numpy() == cfb_label
        )
        if len(cfb_indices) < cfb_target_count:
            raise ValueError(
                f'CFB当前只有 {len(cfb_indices)} 条，'
                f'不能欠采样到 {cfb_target_count}'
            )
        rng = np.random.default_rng(42)
        selected_cfb = np.sort(
            rng.choice(
                cfb_indices,
                size=cfb_target_count,
                replace=False,
            )
        )
        non_cfb_indices = np.flatnonzero(
            df_train['TECTONIC SETTING'].astype(str).to_numpy() != cfb_label
        )
        keep_indices = np.sort(
            np.concatenate([non_cfb_indices, selected_cfb])
        )
        X_train_img = X_train_img[keep_indices]
        X_train_seq = X_train_seq[keep_indices]
        y_train = y_train[keep_indices]
        print(
            f'  CFB欠采样: {len(cfb_indices)} -> {cfb_target_count}，'
            f'训练集总数: {len(df_train)} -> {len(keep_indices)}'
        )

    print(f'\n  训练集: {X_train_img.shape[0]} | 测试集: {X_test_img.shape[0]} | 类别数: {len(unique)}')
    print(f'  矩阵输入形状: {X_train_img.shape[1:]}')
    print(f'  序列输入形状: {X_train_seq.shape[1:]}  (按标准电极电势 E° 排序)')
    print(f'  训练集分布: {dict(sorted(collections.Counter(y_train).items()))}')
    print(f'  测试集分布: {dict(sorted(collections.Counter(y_test).items()))}')
    return X_train_img, X_train_seq, y_train, X_test_img, X_test_seq, y_test, unique


# =============================================================
# Mixup 数据增强
# =============================================================

def mixup_data(x_img, x_seq, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x_img.size(0)).to(x_img.device)
    mixed_img = lam * x_img + (1 - lam) * x_img[index]
    mixed_seq = lam * x_seq + (1 - lam) * x_seq[index]
    return mixed_img, mixed_seq, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================
# 共享模块
# =============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches):
        super().__init__()
        self.proj      = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class PatchEmbeddingNoPos(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
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


class CNNBranch(nn.Module):
    """仅用于对比模型 Cmp-2 (CNN-ViT-Transformer) 和 Cmp-3 (CNN Only)。"""
    def __init__(self, in_channels=1, dropout=0.2):
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


# =============================================================
# 【新主模型】Full Model: ViT-Transformer Dual Stream
# =============================================================

class ViT_Transformer_DualStream(nn.Module):
    """
    增强版双流模型 (v2):
      - 矩阵分支 / 序列分支均增加CLS token，与GAP双路融合
      - 容量适度加宽 (embed_dim 64→96, heads 4→8)
      - 保持原 TransformerBlock(post-norm),不引入未验证的 stochastic depth
      - 分类头融合4路特征: vit_cls + vit_gap + seq_cls + seq_gap
    """
    def __init__(self, num_classes, input_size=6, patch_size=2,
                 embed_dim=128, num_heads=8, transformer_layers=2,
                 ff_dim=256, dropout=0.15, use_missing_mask=False):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2   # 9
        self.seq_len     = input_size * input_size           # 36
        self.embed_dim   = embed_dim
        self.use_missing_mask = use_missing_mask
        image_channels = 2 if use_missing_mask else 1
        sequence_features = 2 if use_missing_mask else 1

        # ── 矩阵分支 (ViT) ────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            image_channels,
            patch_size,
            embed_dim,
            self.num_patches,
        )
        self.vit_cls     = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vit_cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vit_blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.vit_norm    = nn.LayerNorm(embed_dim)

        # ── 序列分支 (Transformer) ────────────────────────────
        self.seq_proj      = nn.Linear(sequence_features, embed_dim)
        self.seq_norm      = nn.LayerNorm(embed_dim)
        self.seq_cls       = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seq_pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len + 1, embed_dim) * 0.02)
        self.seq_blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.seq_final_norm = nn.LayerNorm(embed_dim)

        # ── 分类头: 4路特征 (vit_cls + vit_gap + seq_cls + seq_gap) ──
        head_in = embed_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(head_in, 192),
            nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.LayerNorm(96),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.vit_cls,     std=0.02)
        nn.init.trunc_normal_(self.seq_cls,     std=0.02)
        nn.init.trunc_normal_(self.vit_cls_pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_seq):
        batch_size = x.size(0)

        # 矩阵分支
        vit_tokens = self.patch_embed(x)                       # (B, 9, D)
        vit_cls = self.vit_cls.expand(batch_size, -1, -1)
        vit_cls = vit_cls + self.vit_cls_pos
        vit_tokens = torch.cat([vit_cls, vit_tokens], dim=1)   # (B, 10, D)
        for blk in self.vit_blocks:
            vit_tokens = blk(vit_tokens)
        vit_tokens  = self.vit_norm(vit_tokens)
        vit_cls_out = vit_tokens[:, 0]
        vit_gap_out = vit_tokens[:, 1:].mean(dim=1)

        # 序列分支
        seq_tokens = self.seq_norm(self.seq_proj(x_seq))       # (B, 36, D)
        seq_cls = self.seq_cls.expand(batch_size, -1, -1)
        seq_tokens = torch.cat([seq_cls, seq_tokens], dim=1)   # (B, 37, D)
        seq_tokens = seq_tokens + self.seq_pos_embed
        for blk in self.seq_blocks:
            seq_tokens = blk(seq_tokens)
        seq_tokens  = self.seq_final_norm(seq_tokens)
        seq_cls_out = seq_tokens[:, 0]
        seq_gap_out = seq_tokens[:, 1:].mean(dim=1)

        # 中文注释：恢复CLS与GAP四路融合，以保留两种聚合方式的互补信息。
        fused = torch.cat(
            [vit_cls_out, vit_gap_out, seq_cls_out, seq_gap_out],
            dim=1,
        )
        return self.fusion(fused)


# =============================================================
# 消融 1【新增】: ViT Only (仅矩阵分支)
# =============================================================

class Ablation_ViT_Only(nn.Module):
    """
    Abl-1 新增对比模型：仅使用矩阵分支的 ViT，不使用序列分支。
    回答"序列分支是否必要"。
    """
    def __init__(self, num_classes, input_size=6, patch_size=2,
                 embed_dim=64, num_heads=4, transformer_layers=2,
                 ff_dim=128, dropout=0.2, use_missing_mask=False):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2
        image_channels = 2 if use_missing_mask else 1

        self.patch_embed = PatchEmbedding(
            image_channels, patch_size, embed_dim, self.num_patches
        )
        self.vit_blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_seq=None):
        out = self.patch_embed(x)
        for blk in self.vit_blocks:
            out = blk(out)
        return self.classifier(out.mean(dim=1))


# =============================================================
# 消融 2: Transformer Only (仅序列分支)
# =============================================================

class Ablation_Transformer_Only(nn.Module):
    """Abl-2: 仅使用序列分支的 Transformer，回答'矩阵分支是否必要'。"""
    def __init__(self, num_classes, input_size=6,
                 embed_dim=64, num_heads=4, transformer_layers=2,
                 ff_dim=128, dropout=0.2, use_missing_mask=False):
        super().__init__()
        sequence_features = 2 if use_missing_mask else 1
        self.seq_len       = input_size * input_size
        self.seq_proj      = nn.Linear(sequence_features, embed_dim)
        self.seq_norm      = nn.LayerNorm(embed_dim)
        self.seq_pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len, embed_dim) * 0.02)
        self.seq_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x=None, x_seq=None):
        out = self.seq_norm(self.seq_proj(x_seq))
        out = out + self.seq_pos_embed
        for blk in self.seq_blocks:
            out = blk(out)
        return self.classifier(out.mean(dim=1))


# =============================================================
# 消融 3: 双流 w/o Positional Encoding
# =============================================================

class Ablation_NoPositionalEncoding(nn.Module):
    """
    Abl-3: 在新主模型基础上去掉位置编码（矩阵 ViT 和序列 Transformer 都不加 PE）。
    回答"位置编码是否必要"。在 6×6 元素矩阵中，元素身份与位置严格对应，
    模型可能从内容隐式推断位置。
    """
    def __init__(self, num_classes, input_size=6, patch_size=2,
                 embed_dim=64, num_heads=4, transformer_layers=2,
                 ff_dim=128, dropout=0.2, use_missing_mask=False):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2
        self.seq_len     = input_size * input_size
        image_channels = 2 if use_missing_mask else 1
        sequence_features = 2 if use_missing_mask else 1

        # 矩阵分支: 无 PE 的 PatchEmbedding
        self.patch_embed = PatchEmbeddingNoPos(
            image_channels, patch_size, embed_dim, self.num_patches
        )
        self.vit_blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])

        # 序列分支: 不加 pos_embed
        self.seq_proj   = nn.Linear(sequence_features, embed_dim)
        self.seq_norm   = nn.LayerNorm(embed_dim)
        self.seq_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_seq):
        vit_out = self.patch_embed(x)
        for blk in self.vit_blocks:
            vit_out = blk(vit_out)
        seq_out = self.seq_norm(self.seq_proj(x_seq))
        for blk in self.seq_blocks:
            seq_out = blk(seq_out)
        return self.fusion(torch.cat([vit_out.mean(1), seq_out.mean(1)], dim=1))


# =============================================================
# 对比模型 1: CNN-BiLSTM (EMSPN 前作)
# =============================================================

class ConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.4):
        super().__init__()
        self.conv1   = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(channels)
        self.conv2   = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.bn2(F.relu(self.conv2(out)))
        return self.dropout(out)


class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes, bilstm_hidden=32, dropout=0.5,
                 use_missing_mask=False):
        super().__init__()
        image_channels = 2 if use_missing_mask else 1
        sequence_features = 2 if use_missing_mask else 1
        self.cnn_conv1   = nn.Conv2d(image_channels, 16, 3, padding=1)
        self.cnn_bn1     = nn.BatchNorm2d(16)
        self.cnn_blk1    = ConvBlock(16, dropout=dropout)
        self.cnn_conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.cnn_bn2     = nn.BatchNorm2d(32)
        # cnn_blk2 removed to reduce capacity (bilstm_hidden also reduced 64→48)
        self.cnn_fc      = nn.Linear(32 * 6 * 6, 64)
        self.cnn_fc_bn   = nn.BatchNorm1d(64)
        self.cnn_fc_drop = nn.Dropout(dropout)

        self.seq_proj      = nn.Linear(sequence_features, bilstm_hidden)
        self.seq_proj_bn   = nn.BatchNorm1d(bilstm_hidden)
        self.seq_proj_drop = nn.Dropout(dropout)

        self.bilstm1        = nn.LSTM(bilstm_hidden, bilstm_hidden,
                                      batch_first=True, bidirectional=True)
        self.rnn_bn1        = nn.BatchNorm1d(bilstm_hidden * 2)
        self.rnn_inter_drop = nn.Dropout(dropout)
        self.bilstm2        = nn.LSTM(bilstm_hidden * 2, bilstm_hidden,
                                      batch_first=True, bidirectional=True)
        self.rnn_bn2        = nn.BatchNorm1d(bilstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(64 + bilstm_hidden * 2, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, x_seq):
        c = F.relu(self.cnn_bn1(self.cnn_conv1(x)))
        c = self.cnn_blk1(c)
        c = F.relu(self.cnn_bn2(self.cnn_conv2(c)))
        c = self.cnn_fc_drop(F.relu(self.cnn_fc_bn(self.cnn_fc(c.flatten(1)))))

        B, T, _ = x_seq.shape
        r = self.seq_proj(x_seq.reshape(B * T, -1))
        r = self.seq_proj_drop(F.relu(self.seq_proj_bn(r))).reshape(B, T, -1)
        r, _ = self.bilstm1(r)
        r = self.rnn_bn1(r.transpose(1, 2)).transpose(1, 2)
        r = self.rnn_inter_drop(r)
        r, _ = self.bilstm2(r)
        r = self.rnn_bn2(r[:, -1, :])
        return self.classifier(torch.cat([c, r], dim=1))


# =============================================================
# 对比模型 2: CNN-ViT-Transformer (旧 Full Model)
# =============================================================

class CNN_ViT_Transformer(nn.Module):
    """
    Cmp-2: 旧版 Full Model，含 CNN backbone。
    与新主模型对比，证明 CNN 在 6×6 输入上是冗余的。
    """
    def __init__(self, num_classes, input_size=6, patch_size=2,
                 embed_dim=64, num_heads=4, transformer_layers=2,
                 ff_dim=128, dropout=0.1, use_missing_mask=False):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2
        self.seq_len     = input_size * input_size
        image_channels = 2 if use_missing_mask else 1
        sequence_features = 2 if use_missing_mask else 1

        self.cnn         = CNNBranch(
            in_channels=image_channels, dropout=0.2
        )
        self.patch_embed = PatchEmbedding(64, patch_size, embed_dim, self.num_patches)
        self.vit_blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.seq_proj      = nn.Linear(sequence_features, embed_dim)
        self.seq_norm      = nn.LayerNorm(embed_dim)
        self.seq_pos_embed = nn.Parameter(torch.randn(1, self.seq_len, embed_dim) * 0.02)
        self.seq_blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_seq):
        cnn_feat = self.cnn(x)
        vit_out  = self.patch_embed(cnn_feat)
        for blk in self.vit_blocks:
            vit_out = blk(vit_out)
        seq_out = self.seq_norm(self.seq_proj(x_seq))
        seq_out = seq_out + self.seq_pos_embed
        for blk in self.seq_blocks:
            seq_out = blk(seq_out)
        return self.fusion(torch.cat([vit_out.mean(1), seq_out.mean(1)], dim=1))


# =============================================================
# 对比模型 3: CNN Only
# =============================================================

class Baseline_CNN_Only(nn.Module):
    def __init__(self, num_classes, dropout=0.3, use_missing_mask=False):
        super().__init__()
        image_channels = 2 if use_missing_mask else 1
        self.cnn = CNNBranch(
            in_channels=image_channels, dropout=0.4
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, x_seq=None):
        return self.classifier(self.cnn(x).mean(dim=[2, 3]))


# =============================================================
# 训练工具
# =============================================================

class BestModelTracker:
    def __init__(self):
        self.best_loss  = None
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_model = copy.deepcopy(model.state_dict())


class BestAccTracker:
    def __init__(self):
        self.best_acc   = None
        self.best_epoch = None
        self.best_model = None

    def __call__(self, val_acc, epoch, model):
        if self.best_acc is None or val_acc > self.best_acc:
            self.best_acc   = val_acc
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model.state_dict())


def make_dataloader(X_img, X_seq, y, batch_size, shuffle, drop_last=False):
    return DataLoader(
        TensorDataset(
            torch.FloatTensor(X_img),
            torch.FloatTensor(X_seq),
            torch.LongTensor(y),
        ),
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0,
    )


def train_epoch(model, dataloader, criterion, optimizer, device,
                epoch, total_epochs, mixup_alpha=0):
    model.train()
    total_loss = correct = total = 0
    # 中文注释：每个epoch显示批次训练进度，完整指标仍按PRINT_EVERY间隔打印。
    pbar = tqdm(
        dataloader,
        desc=f'  Epoch {epoch}/{total_epochs}',
        leave=False,
        ncols=100,
        disable=False,
    )
    for X_img, X_seq, y in pbar:
        X_img, X_seq, y = X_img.to(device), X_seq.to(device), y.to(device)
        mixed_img, mixed_seq, y_a, y_b, lam = mixup_data(
            X_img, X_seq, y, alpha=mixup_alpha)
        optimizer.zero_grad()
        outputs = model(mixed_img, mixed_seq)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += y.size(0)
        correct += (lam * predicted.eq(y_a).sum().item()
                    + (1 - lam) * predicted.eq(y_b).sum().item())
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_img, X_seq, y in dataloader:
            X_img, X_seq, y = X_img.to(device), X_seq.to(device), y.to(device)
            outputs = model(X_img, X_seq)
            loss    = criterion(outputs, y)
            probs   = F.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            total_loss += loss.item()
            total      += y.size(0)
            correct    += pred.eq(y).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return (total_loss / len(dataloader), correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


# =============================================================
# 单次（单种子）实验
# =============================================================

def run_single_experiment(model_factory, model_name,
                          X_train_img, X_train_seq, y_train,
                          X_val_img,   X_val_seq,   y_val,
                          num_classes, device,
                          epochs=200, mixup_alpha=0,
                          seed=42):
    """运行单个消融/对比实验（单个种子）。"""
    set_seed(seed)
    model      = model_factory().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"=" * 65}')
    print(f'  实验: {model_name}  |  seed={seed}')
    print(f'  参数量: {num_params:,}')
    print(f'{"=" * 65}')

    t_start = time.time()

    train_loader = make_dataloader(
        X_train_img, X_train_seq, y_train, 64, shuffle=True, drop_last=True)
    val_loader   = make_dataloader(
        X_val_img,   X_val_seq,   y_val,   64, shuffle=False, drop_last=False)

    if USE_CLASS_WEIGHTED_LOSS:
        # 中文注释：权重只由当前训练集标签计算，不读取测试集分布。
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_weights = len(y_train) / (num_classes * class_counts)
        class_weights = class_weights / class_weights.mean()
        class_weights_tensor = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f'  类别加权交叉熵: 开启，权重={np.round(class_weights, 4).tolist()}')
    else:
        criterion = nn.CrossEntropyLoss()
        print('  类别加权交叉熵: 关闭')
    target_lr     = 3e-4
    warmup_epochs = 10
    optimizer = AdamW(model.parameters(), lr=target_lr, weight_decay=0)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3,
                                      patience=15, verbose=True)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=15)

    loss_tracker = BestModelTracker()
    acc_tracker  = BestAccTracker()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    # 中文注释：断点文件直接在代码中配置；留空表示从头开始训练。
    checkpoint_path = ''
    start_epoch = 1

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if int(checkpoint.get('seed', seed)) == seed:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            history = checkpoint['history']
            acc_tracker.best_acc = checkpoint['best_acc']
            acc_tracker.best_epoch = checkpoint['best_epoch']
            acc_tracker.best_model = checkpoint['best_model']
            loss_tracker.best_loss = checkpoint['best_loss']
            loss_tracker.best_model = checkpoint['best_loss_model']
            start_epoch = int(checkpoint['epoch']) + 1
            print(
                f'  [断点续训] 从 Epoch {start_epoch}/{epochs} 继续: '
                f'{checkpoint_path}'
            )

    for epoch in range(start_epoch, epochs + 1):
        if epoch <= warmup_epochs:
            warmup_lr = target_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer,
                                      device, epoch, epochs, mixup_alpha=mixup_alpha)
        va_loss, va_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        if epoch > warmup_epochs:
            scheduler.step(va_loss)
        loss_tracker(va_loss, model)
        acc_tracker(va_acc, epoch, model)

        # ── v4: 每 20 个 epoch 打印一次（首末 epoch 也打印） ───────
        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == epochs:
            current_lr = optimizer.param_groups[0]['lr']
            best_flag  = ' ★' if epoch == acc_tracker.best_epoch else ''
            print(f'  Epoch {epoch:3d}/{epochs} | LR: {current_lr:.4g} | '
                  f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
                  f'Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}{best_flag}')

        if checkpoint_path:
            checkpoint_directory = os.path.dirname(checkpoint_path)
            if checkpoint_directory:
                os.makedirs(checkpoint_directory, exist_ok=True)
            torch.save(
                {
                    'epoch': epoch,
                    'seed': seed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'best_acc': acc_tracker.best_acc,
                    'best_epoch': acc_tracker.best_epoch,
                    'best_model': acc_tracker.best_model,
                    'best_loss': loss_tracker.best_loss,
                    'best_loss_model': loss_tracker.best_model,
                },
                checkpoint_path,
            )

    model.load_state_dict(acc_tracker.best_model)
    va_loss, va_acc, y_pred, y_probs, y_true = evaluate(
        model, val_loader, criterion, device)

    elapsed = time.time() - t_start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f'\n  [完成] seed={seed}，耗时: {h:02d}h {m:02d}m {s:02d}s')
    print(f'  最高验证准确率: Epoch {acc_tracker.best_epoch}  Val Acc={acc_tracker.best_acc:.4f}')

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(   y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(       y_true, y_pred, average='weighted', zero_division=0)
    macro_f1  = f1_score(       y_true, y_pred, average='macro',    zero_division=0)
    y_onehot  = np.eye(num_classes)[y_true]
    mAP       = np.mean([average_precision_score(y_onehot[:, i], y_probs[:, i])
                          for i in range(num_classes)])

    print(f'  最终结果 | Acc: {va_acc:.4f} | F1: {f1:.4f} | Macro F1: {macro_f1:.4f} | mAP: {mAP:.4f}')
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        'model_name':      model_name,
        'num_params':      num_params,
        'seed':            seed,
        'accuracy':        va_acc,
        'precision':       precision,
        'recall':          recall,
        'f1_score':        f1,
        'macro_f1':         macro_f1,
        'mAP':             mAP,
        'val_loss':        va_loss,
        'best_acc_epoch':  acc_tracker.best_epoch,
        'y_true':          y_true,
        'y_pred':          y_pred,
        'y_probs':         y_probs,
        'history':         history,
        'best_state_dict': acc_tracker.best_model,
        'elapsed_sec':     elapsed,
    }


# =============================================================
# 多种子包装：对给定实验跑所有 SEEDS，聚合 mean ± std
# =============================================================

def run_experiment_multi_seed(model_factory, model_name,
                              X_train_img, X_train_seq, y_train,
                              X_val_img,   X_val_seq,   y_val,
                              num_classes, device,
                              epochs=200, mixup_alpha=0,
                              seeds=None,
                              output_dir='.',
                              save_per_seed_weights=True):
    if seeds is None:
        seeds = SEEDS

    n_seeds   = len(seeds)
    fname_base = safe_filename(model_name)
    seed_dir   = os.path.join(output_dir, 'per_seed_weights', fname_base)
    if save_per_seed_weights:
        os.makedirs(seed_dir, exist_ok=True)

    print(f'\n{"#" * 70}')
    print(f'  多种子实验: {model_name}')
    print(f'  Seeds = {seeds}  (共 {n_seeds} 次)')
    print(f'{"#" * 70}')

    per_seed_results = []
    total_t0 = time.time()

    for i, seed in enumerate(seeds):
        print(f'\n  ── 种子 {seed} ({i + 1}/{n_seeds}) ──')
        result = run_single_experiment(
            model_factory, model_name,
            X_train_img, X_train_seq, y_train,
            X_val_img,   X_val_seq,   y_val,
            num_classes, device,
            epochs=epochs, mixup_alpha=mixup_alpha,
            seed=seed,
        )
        per_seed_results.append(result)

        if save_per_seed_weights and result['best_state_dict'] is not None:
            w_path = os.path.join(seed_dir, f'seed{seed}.pth')
            torch.save(result['best_state_dict'], w_path)
            print(f'  权重已保存: {w_path}')

    # ── 聚合 ────────────────────────────────────────────────────────
    # 中文注释：保留加权 F1，同时新增 Macro F1，便于观察小类别表现。
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'macro_f1', 'mAP', 'val_loss']
    agg = {}
    for k in metric_keys:
        vals       = [r[k] for r in per_seed_results]
        agg[k]          = float(np.mean(vals))
        agg[f'{k}_std'] = float(np.std(vals, ddof=1))   # 样本标准差

    best_seed_idx = int(np.argmax([r['accuracy'] for r in per_seed_results]))
    best_seed_r   = per_seed_results[best_seed_idx]

    total_elapsed = time.time() - total_t0
    th, tm, ts = (int(total_elapsed // 3600),
                  int((total_elapsed % 3600) // 60),
                  int(total_elapsed % 60))

    print(f'\n{"=" * 65}')
    print(f'  [完成] {model_name}  多种子汇总 ({n_seeds} 个种子)')
    print(f'    Accuracy : {agg["accuracy"]:.4f} ± {agg["accuracy_std"]:.4f}')
    print(f'    F1-Score : {agg["f1_score"]:.4f} ± {agg["f1_score_std"]:.4f}')
    print(f'    Macro F1 : {agg["macro_f1"]:.4f} ± {agg["macro_f1_std"]:.4f}')
    print(f'    mAP      : {agg["mAP"]:.4f} ± {agg["mAP_std"]:.4f}')
    print(f'    最优种子 : {seeds[best_seed_idx]}  '
          f'(Acc={best_seed_r["accuracy"]:.4f})')
    print(f'    总耗时   : {th:02d}h {tm:02d}m {ts:02d}s')
    print(f'{"=" * 65}')

    return {
        'model_name':         model_name,
        'num_params':         per_seed_results[0]['num_params'],
        'accuracy':           best_seed_r['accuracy'],
        'precision':          best_seed_r['precision'],
        'recall':             best_seed_r['recall'],
        'f1_score':           best_seed_r['f1_score'],
        'macro_f1':           best_seed_r['macro_f1'],
        'mAP':                best_seed_r['mAP'],
        'val_loss':           best_seed_r['val_loss'],
        'best_acc_epoch':     best_seed_r['best_acc_epoch'],
        'y_true':             best_seed_r['y_true'],
        'y_pred':             best_seed_r['y_pred'],
        'y_probs':            best_seed_r['y_probs'],
        'history':            best_seed_r['history'],
        'best_state_dict':    best_seed_r['best_state_dict'],
        'elapsed_sec':        total_elapsed,
        'accuracy_std':       agg['accuracy_std'],
        'precision_std':      agg['precision_std'],
        'recall_std':         agg['recall_std'],
        'f1_score_std':       agg['f1_score_std'],
        'macro_f1_std':       agg['macro_f1_std'],
        'mAP_std':            agg['mAP_std'],
        'val_loss_std':       agg['val_loss_std'],
        'per_seed_results':   per_seed_results,
        'seeds':              seeds,
        'best_seed':          seeds[best_seed_idx],
        'seed_mean_accuracy': agg['accuracy'],
        'seed_mean_macro_f1': agg['macro_f1'],
    }


# =============================================================
# 传统 ML 基线
# =============================================================

_ML_XGB_EPOCHS   = 200
_ML_MLP_EPOCHS   = 200
_ML_CURVE_MODELS = {'XGBoost', 'MLP'}


def _ml_label_mapping():
    return {
        'Continental arc':          'CA',
        'Island arc':               'IA',
        'Intra-oceanic arc':        'IOA',
        'BACK-ARC_BASIN':           'BAB',
        'SPREADING_CENTER':         'MOR',
        'OCEANIC PLATEAU':          'OP',
        'OCEAN ISLAND':             'OI',
        'CONTINENTAL FLOOD BASALT': 'CF',
        'CONTINENTAL_RIFT':         'CR',
    }


def _ml_build_random_forest(seed=42):
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=25,
        min_samples_leaf=15, max_features='sqrt', max_samples=0.65,
        bootstrap=True, random_state=seed, n_jobs=-1, verbose=0,
    )


def _ml_build_svm(seed=42):
    return SVC(C=0.1, kernel='rbf', gamma='scale', probability=True,
               random_state=seed, cache_size=2000, verbose=False)


def _ml_build_xgboost(num_classes, seed=42):
    if XGBClassifier is None:
        raise ImportError('xgboost 未安装，请 pip install xgboost')
    return XGBClassifier(
        n_estimators=_ML_XGB_EPOCHS, max_depth=4, learning_rate=0.05,
        subsample=0.65, colsample_bytree=0.7, min_child_weight=10,
        gamma=1.0, reg_alpha=3.0, reg_lambda=5.0,
        objective='multi:softprob', num_class=num_classes,
        eval_metric=['mlogloss', 'merror'], use_label_encoder=False,
        random_state=seed, n_jobs=-1, verbosity=0,
    )


def _ml_build_mlp(seed=42):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
        alpha=0.15, batch_size=64, max_iter=200, early_stopping=False,
        validation_fraction=0.1, n_iter_no_change=20,
        learning_rate='adaptive', learning_rate_init=5e-4,
        random_state=seed, verbose=False, warm_start=True,
    )


def _ml_get_all_models(num_classes, seed=42):
    models = {
        'RandomForest': _ml_build_random_forest(seed),
        'SVM':          _ml_build_svm(seed),
    }
    if XGBClassifier is not None:
        models['XGBoost'] = _ml_build_xgboost(num_classes, seed)
    else:
        print('[WARNING] xgboost 未安装，跳过 XGBoost 基线。')
    models['MLP'] = _ml_build_mlp(seed)
    return models


def _ml_evaluate_model(model, X_test, y_test, num_classes):
    y_pred  = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    y_onehot = label_binarize(y_test, classes=np.arange(num_classes))
    per_ap = [
        average_precision_score(y_onehot[:, i], y_probs[:, i])
        if y_onehot[:, i].sum() > 0 else 0.0
        for i in range(num_classes)
    ]
    return {
        'accuracy':  np.mean(y_pred == y_test),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(   y_test, y_pred, average='weighted', zero_division=0),
        'f1_score':  f1_score(       y_test, y_pred, average='weighted', zero_division=0),
        'macro_f1':   f1_score(       y_test, y_pred, average='macro',    zero_division=0),
        'mAP':       float(np.mean(per_ap)),
        'y_true':    y_test,
        'y_pred':    y_pred,
        'y_probs':   y_probs,
    }


def _ml_compute_split_metrics(model, X, y, num_classes):
    y_pred  = model.predict(X)
    y_probs = model.predict_proba(X)
    return {
        'loss':     log_loss(y, y_probs, labels=np.arange(num_classes)),
        'accuracy': accuracy_score(y, y_pred),
    }


def _ml_init_history():
    return {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}


def _ml_update_history(history, tr, te):
    history['train_loss'].append(tr['loss'])
    history['test_loss'].append(te['loss'])
    history['train_acc'].append(tr['accuracy'])
    history['test_acc'].append(te['accuracy'])


def _ml_print_epoch(name, epoch, total, tr, te):
    print(f'[{name}] Epoch {epoch:3d}/{total} | '
          f'Train Loss: {tr["loss"]:.4f} Acc: {tr["accuracy"]:.4f} | '
          f'Test  Loss: {te["loss"]:.4f} Acc: {te["accuracy"]:.4f}')


def _ml_train_xgboost(model, X_train, y_train, X_test, y_test):
    """XGBoost 训练；evals_result 记录每一 epoch，但本函数不打印逐 epoch。"""
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)
    res = model.evals_result()
    history = {
        'train_loss': list(res['validation_0']['mlogloss']),
        'test_loss':  list(res['validation_1']['mlogloss']),
        'train_acc':  [1.0 - e for e in res['validation_0']['merror']],
        'test_acc':   [1.0 - e for e in res['validation_1']['merror']],
    }
    # ── v4: XGBoost 也按每 20 epoch 打印一次 ────────────────────
    n_iter = len(history['train_loss'])
    for epoch in range(1, n_iter + 1):
        if epoch % ML_PRINT_EVERY == 0 or epoch == 1 or epoch == n_iter:
            tr = {'loss':     history['train_loss'][epoch - 1],
                  'accuracy': history['train_acc'][epoch - 1]}
            te = {'loss':     history['test_loss'][epoch - 1],
                  'accuracy': history['test_acc'][epoch - 1]}
            _ml_print_epoch('XGBoost', epoch, n_iter, tr, te)
    return model, history


def _ml_train_mlp(model, X_train, y_train, X_test, y_test, num_classes):
    history = _ml_init_history()
    classes = np.arange(num_classes)
    for epoch in range(1, _ML_MLP_EPOCHS + 1):
        if epoch == 1:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)
        tr = _ml_compute_split_metrics(model, X_train, y_train, num_classes)
        te = _ml_compute_split_metrics(model, X_test,  y_test,  num_classes)
        _ml_update_history(history, tr, te)
        # ── v4: MLP 每 20 个 epoch 打印一次（首末也打印） ───────
        if epoch % ML_PRINT_EVERY == 0 or epoch == 1 or epoch == _ML_MLP_EPOCHS:
            _ml_print_epoch('MLP', epoch, _ML_MLP_EPOCHS, tr, te)
    return model, history


def _ml_fit_with_history(name, model, X_train, y_train, X_test, y_test, num_classes):
    if name == 'XGBoost':
        return _ml_train_xgboost(model, X_train, y_train, X_test, y_test)
    if name == 'MLP':
        return _ml_train_mlp(model, X_train, y_train, X_test, y_test, num_classes)
    raise ValueError(f'_ml_fit_with_history 不支持: {name}')


def _ml_plot_confusion_matrix(cm, output_dir, classes, model_name, label_mapping,
                               normalize=True):
    classes_mapped = [label_mapping.get(c, c) for c in classes]
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm.astype(float), row_sum,
            out=np.zeros_like(cm, dtype=float),
            where=row_sum != 0,
        ) * 100.0
    save_path = os.path.join(output_dir, f'confusion_matrix_{model_name}_soft_geodan.png')
    _plot_geodan_confusion_matrix(cm, classes_mapped, save_path)


def _ml_plot_pr_curve(y_onehot, y_probs, output_dir, model_name,
                      unique_labels, label_mapping):
    plt.figure(figsize=(10, 8))
    for i in range(y_onehot.shape[1]):
        if y_onehot[:, i].sum() > 0:
            prec, rec, _ = precision_recall_curve(y_onehot[:, i], y_probs[:, i])
            ap = average_precision_score(y_onehot[:, i], y_probs[:, i])
            label = label_mapping.get(unique_labels[i], unique_labels[i])
            plt.plot(rec, prec, lw=2, label=f'{label} (AP={ap:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, f'pr_curve_{model_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _ml_plot_training_curves(history, output_dir, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=1.5)
    axes[0].plot(epochs, history['test_loss'],  label='Test Loss',  linewidth=1.5)
    axes[0].set_title(f'{model_name} - Loss Curve')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=1.5)
    axes[1].plot(epochs, history['test_acc'],  label='Test Acc',  linewidth=1.5)
    axes[1].set_title(f'{model_name} - Accuracy Curve')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{model_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _ml_save_training_history(history, output_dir, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    pd.DataFrame({
        'epoch':          list(epochs),
        'train_loss':     history['train_loss'],
        'test_loss':      history['test_loss'],
        'train_accuracy': history['train_acc'],
        'test_accuracy':  history['test_acc'],
    }).to_csv(os.path.join(output_dir, f'training_history_{model_name}.csv'), index=False)


def _ml_train_and_evaluate(model_name, model,
                            X_train, y_train, X_test, y_test,
                            num_classes, unique_labels, label_mapping, output_dir):
    print(f'\n{"=" * 60}\n  ML 基线训练: {model_name}\n{"=" * 60}')
    t0 = time.time()

    if model_name in _ML_CURVE_MODELS:
        model, history = _ml_fit_with_history(
            model_name, model, X_train, y_train, X_test, y_test, num_classes)
        train_time = time.time() - t0
        result = _ml_evaluate_model(model, X_test, y_test, num_classes)
        result['train_time']     = train_time
        result['epochs_ran']     = len(history['train_loss'])
        result['train_loss']     = history['train_loss'][-1]
        result['train_accuracy'] = history['train_acc'][-1]
        result['test_loss']      = history['test_loss'][-1]
        _ml_plot_training_curves(history, output_dir, model_name)
        _ml_save_training_history(history, output_dir, model_name)
    else:
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        result = _ml_evaluate_model(model, X_test, y_test, num_classes)
        result['train_time'] = train_time
        tr = _ml_compute_split_metrics(model, X_train, y_train, num_classes)
        te = _ml_compute_split_metrics(model, X_test,  y_test,  num_classes)
        result['epochs_ran']     = 1
        result['train_loss']     = tr['loss']
        result['train_accuracy'] = tr['accuracy']
        result['test_loss']      = te['loss']

    result['model'] = model_name
    print(f'  训练耗时: {train_time:.1f}s')
    print(f'  Train Acc: {result["train_accuracy"]:.4f} | '
          f'Test Acc: {result["accuracy"]:.4f} | '
          f'F1: {result["f1_score"]:.4f} | '
          f'Macro F1: {result["macro_f1"]:.4f} | mAP: {result["mAP"]:.4f}')
    print(classification_report(result['y_true'], result['y_pred'], zero_division=0))

    # 中文注释：固定 labels，确保 ML 混淆矩阵维度与类别标签一一对应。
    cm = confusion_matrix(result['y_true'], result['y_pred'], labels=np.arange(num_classes))
    _ml_plot_confusion_matrix(cm, output_dir, unique_labels, model_name, label_mapping)
    y_onehot = label_binarize(result['y_true'], classes=np.arange(num_classes))
    _ml_plot_pr_curve(y_onehot, result['y_probs'],
                      output_dir, model_name, unique_labels, label_mapping)
    return result


def _as_ml_features(X_img):
    return X_img.reshape(X_img.shape[0], -1).astype(np.float32)


def run_ml_baseline_experiments(X_train_img, y_train,
                                X_test_img, y_test,
                                num_classes, unique_labels, output_dir,
                                seeds=None):
    if seeds is None:
        seeds = SEEDS

    X_train_ml = _as_ml_features(X_train_img)
    X_test_ml  = _as_ml_features(X_test_img)
    label_mapping = _ml_label_mapping()
    ml_output_dir = os.path.join(output_dir, 'ML_Baselines')
    os.makedirs(ml_output_dir, exist_ok=True)

    print('\n' + '=' * 80)
    print(f'传统 ML 基线（多种子，seeds={seeds}）')
    print('=' * 80)

    all_models_seed0 = _ml_get_all_models(num_classes, seed=seeds[0])
    model_names = list(all_models_seed0.keys())

    ml_results = []
    for model_name in model_names:
        seed_metrics = {k: [] for k in
                        ['accuracy', 'precision', 'recall', 'f1_score', 'macro_f1', 'mAP', 'test_loss']}
        representative_result = None

        for seed in seeds:
            print(f'\n  ML [{model_name}]  seed={seed}')
            models_this_seed = _ml_get_all_models(num_classes, seed=seed)
            model = models_this_seed[model_name]

            if seed == seeds[0]:
                r = _ml_train_and_evaluate(
                    model_name, model,
                    X_train_ml, y_train, X_test_ml, y_test,
                    num_classes, unique_labels, label_mapping, ml_output_dir,
                )
                representative_result = r
            else:
                if model_name in _ML_CURVE_MODELS:
                    model, _ = _ml_fit_with_history(
                        model_name, model, X_train_ml, y_train,
                        X_test_ml, y_test, num_classes)
                else:
                    model.fit(X_train_ml, y_train)
                r = _ml_evaluate_model(model, X_test_ml, y_test, num_classes)
                te = _ml_compute_split_metrics(model, X_test_ml, y_test, num_classes)
                r['test_loss'] = te['loss']

            seed_metrics['accuracy'].append(r['accuracy'])
            seed_metrics['precision'].append(r['precision'])
            seed_metrics['recall'].append(r['recall'])
            seed_metrics['f1_score'].append(r['f1_score'])
            seed_metrics['macro_f1'].append(r['macro_f1'])
            seed_metrics['mAP'].append(r['mAP'])
            seed_metrics['test_loss'].append(r.get('test_loss', 0.0))

        agg_acc  = float(np.mean(seed_metrics['accuracy']))
        agg_f1   = float(np.mean(seed_metrics['f1_score']))
        agg_macro_f1 = float(np.mean(seed_metrics['macro_f1']))
        agg_map  = float(np.mean(seed_metrics['mAP']))
        std_acc  = float(np.std(seed_metrics['accuracy'],  ddof=1))
        std_f1   = float(np.std(seed_metrics['f1_score'],  ddof=1))
        std_macro_f1 = float(np.std(seed_metrics['macro_f1'], ddof=1))
        std_map  = float(np.std(seed_metrics['mAP'],       ddof=1))

        print(f'\n  ML [{model_name}] 多种子汇总:  '
              f'Acc={agg_acc:.4f}±{std_acc:.4f}  '
              f'F1={agg_f1:.4f}±{std_f1:.4f}  '
              f'Macro F1={agg_macro_f1:.4f}±{std_macro_f1:.4f}  '
              f'mAP={agg_map:.4f}±{std_map:.4f}')

        history = {
            'train_loss': [representative_result.get('train_loss', 0.0)],
            'val_loss':   [representative_result.get('test_loss', 0.0)],
            'train_acc':  [representative_result.get('train_accuracy', 0.0)],
            'val_acc':    [representative_result['accuracy']],
        }

        ml_results.append({
            'model_name':    f'ML\n{model_name}',
            'num_params':    'N/A',
            'accuracy':      agg_acc,
            'precision':     float(np.mean(seed_metrics['precision'])),
            'recall':        float(np.mean(seed_metrics['recall'])),
            'f1_score':      agg_f1,
            'macro_f1':      agg_macro_f1,
            'mAP':           agg_map,
            'val_loss':      float(np.mean(seed_metrics['test_loss'])),
            'accuracy_std':  std_acc,
            'precision_std': float(np.std(seed_metrics['precision'], ddof=1)),
            'recall_std':    float(np.std(seed_metrics['recall'],    ddof=1)),
            'f1_score_std':  std_f1,
            'macro_f1_std':  std_macro_f1,
            'mAP_std':       std_map,
            'val_loss_std':  float(np.std(seed_metrics['test_loss'], ddof=1)),
            'y_true':        representative_result['y_true'],
            'y_pred':        representative_result['y_pred'],
            'y_probs':       representative_result['y_probs'],
            'history':       history,
            'best_state_dict': None,
            'elapsed_sec':   0.0,
            'best_acc_epoch': 'N/A',
            'seeds':         seeds,
        })

    pd.DataFrame([{
        'model':         r['model_name'].replace('\n', ' '),
        'accuracy_mean': r['accuracy'],
        'accuracy_std':  r.get('accuracy_std', 0),
        'f1_mean':       r['f1_score'],
        'f1_std':        r.get('f1_score_std', 0),
        'macro_f1_mean':  r['macro_f1'],
        'macro_f1_std':   r.get('macro_f1_std', 0),
        'mAP_mean':      r['mAP'],
        'mAP_std':       r.get('mAP_std', 0),
    } for r in ml_results]).to_csv(
        os.path.join(ml_output_dir, 'ML_baselines_summary.csv'),
        index=False, encoding='utf-8-sig',
    )

    return ml_results


# =============================================================
# 结果可视化（带 mean ± std error bar）
# =============================================================

def plot_ablation_comparison(all_results, output_dir, unique_labels, label_mapping):
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    class_names = [label_mapping.get(label, label) for label in unique_labels]
    class_ids = np.arange(len(unique_labels))

    for r in all_results:
        cm      = confusion_matrix(r['y_true'], r['y_pred'], labels=class_ids)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct  = np.divide(
            cm, row_sum,
            out=np.zeros_like(cm, dtype=float),
            where=row_sum != 0,
        ) * 100.0
        fname = safe_filename(r['model_name'])
        save_path = os.path.join(cm_dir, f'cm_{fname}_soft_geodan.png')
        _plot_geodan_confusion_matrix(cm_pct, class_names, save_path)
        print(f'  混淆矩阵已保存: {save_path}')

    names   = [r['model_name'].replace('\n', '\n') for r in all_results]
    metric_keys   = ['accuracy', 'f1_score', 'macro_f1', 'precision', 'recall', 'mAP']
    metric_labels = ['Accuracy', 'Weighted F1', 'Macro F1', 'Precision', 'Recall', 'mAP']
    values  = {k: [r[k] for r in all_results] for k in metric_keys}
    stds    = {k: [r.get(f'{k}_std', 0.0) for r in all_results] for k in metric_keys}

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
              '#7f8c8d', '#c0392b', '#16a085', '#8e44ad']
    fig, axes = plt.subplots(1, 6, figsize=(32, 6))
    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        vals = values[mk]
        errs = stds[mk]
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]
        bars = ax.bar(range(len(names)), vals,
                      yerr=errs,
                      color=bar_colors, edgecolor='white', linewidth=0.5,
                      error_kw=dict(ecolor='#2c3e50', capsize=3,
                                    elinewidth=1.2, capthick=1.2))
        ax.set_title(ml, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        lo = max(0.0, min(v - e for v, e in zip(vals, errs)) - 0.08)
        ax.set_ylim(lo, 1.05)
        for bar, val, err in zip(bars, vals, errs):
            label = f'{val:.4f}' if err == 0 else f'{val:.4f}\n±{err:.4f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(err, 0) + 0.004,
                    label, ha='center', va='bottom', fontsize=6)
        ax.grid(axis='y', alpha=0.3)

    n_seeds = len(SEEDS)
    plt.suptitle(
        f'Ablation Study — ViT-Transformer Dual Stream 各模块贡献对比  '
        f'(mean ± std, {n_seeds} seeds)',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n对比图已保存: {save_path}')


def plot_all_training_curves(all_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for r in all_results:
        h     = r['history']
        label = r['model_name'].replace('\n', ' ')
        axes[0].plot(h['val_loss'], label=label, alpha=0.85)
        axes[1].plot(h['val_acc'],  label=label, alpha=0.85)

    for ax, title, ylabel in zip(
        axes,
        ['Validation Loss (best-seed curve)', 'Validation Accuracy (best-seed curve)'],
        ['Loss', 'Accuracy'],
    ):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Ablation Study — 最优种子训练过程对比',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, 'ablation_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'训练曲线已保存: {save_path}')


def plot_individual_training_curves(all_results, output_dir):
    curves_dir = os.path.join(output_dir, 'training_curves_by_model')
    os.makedirs(curves_dir, exist_ok=True)

    for r in all_results:
        h           = r['history']
        epochs      = range(1, len(h['train_loss']) + 1)
        model_label = r['model_name'].replace('\n', ' ')
        best_seed   = r.get('best_seed', '?')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, h['train_loss'], label='Train Loss', linewidth=1.8)
        axes[0].plot(epochs, h['val_loss'],   label='Test Loss',  linewidth=1.8)
        axes[0].set_title('Loss', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, h['train_acc'], label='Train Accuracy', linewidth=1.8)
        axes[1].plot(epochs, h['val_acc'],   label='Test Accuracy',  linewidth=1.8)
        axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1.02)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

        plt.suptitle(f'Training/Test Curves - {model_label}  (seed={best_seed})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname     = safe_filename(r['model_name'])
        save_path = os.path.join(curves_dir, f'train_test_curves_{fname}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'单模型训练/测试曲线已保存: {save_path}')


def plot_roc_pr_sci_comparison(all_results, output_dir,
                                models_to_compare=None,
                                figsize=(13, 5.2)):
    """
    SCI 高水平期刊风格的 ROC + PR 对比图（双图合并，左 a 右 b）。

    样式参考典型 Earth-Science / Chemical Geology 文章：
      - 白底 / 浅灰网格
      - 6 模型 micro-averaged ROC + PR 曲线
      - 图例带 AUC / mAP 数值
      - a/b 子图标签

    参数:
        all_results        : 所有实验结果字典列表（DL + ML）
        output_dir         : 输出目录
        models_to_compare  : 要对比的模型 model_name 列表；
                             None 则自动挑选 Full + CNN-BiLSTM + ML 4 个基线
        figsize            : 图尺寸
    """
    from sklearn.metrics import (roc_curve, auc,
                                 precision_recall_curve as _pr_curve,
                                 roc_auc_score, average_precision_score as _ap)

    # ── 默认对比模型：主模型 + CNN-BiLSTM + 4 个 ML 基线 ────────
    if models_to_compare is None:
        candidates_main = [r for r in all_results
                           if r['model_name'].startswith('Full Model')]
        candidates_bilstm = [r for r in all_results
                             if 'CNN-BiLSTM' in r['model_name']]
        candidates_cnn = [r for r in all_results
                          if 'CNN Only' in r['model_name']]
        candidates_ml = {n: None for n in ['RandomForest', 'SVM', 'XGBoost', 'MLP']}
        for r in all_results:
            for ml_name in candidates_ml:
                if ml_name in r['model_name']:
                    candidates_ml[ml_name] = r
        ordered = (candidates_main
                   + candidates_bilstm
                   + candidates_cnn
                   + [candidates_ml['XGBoost'],
                      candidates_ml['MLP'],
                      candidates_ml['SVM'],
                      candidates_ml['RandomForest']])
        ordered = [r for r in ordered if r is not None]
    else:
        ordered = [r for r in all_results
                   if r['model_name'].replace('\n', ' ').strip() in models_to_compare
                   or any(name in r['model_name'] for name in models_to_compare)]

    if len(ordered) < 2:
        print('[WARN] plot_roc_pr_sci_comparison: 可用模型不足，跳过对比图')
        return

    # ── SCI 期刊配色 ──────────────────────────────────────────
    style_map = [
        ('Full',         '#C0392B', 1.5, '-'),    # 主模型 EMSAN: 深红（视觉焦点）
        ('CNN-BiLSTM',   '#E67E22', 1.5, '-'),    # EMSPN: 深橙（同源区分）
        ('CNN Only',     '#8E44AD', 1.5, '-'),    # 纯 CNN: 紫色
        ('MLP',          '#16A085', 1.5, '-'),    # MLP: 青绿
        ('SVM',          '#566573', 1.5, '-'),    # SVM: 深灰
        ('RandomForest', '#7F8C4E', 1.5, '-'),    # RF: 橄榄绿
        ('XGBoost',      '#3498DB', 1.5, '-'),    # XGBoost: 蓝
    ]

    def _pretty_label(model_name: str) -> str:
        # 中文注释：ROC/PR 对比图使用论文中的简写模型名。
        s = model_name.replace('\n', ' ').replace('Full Model', 'GeoDAN')
        s = s.replace('(ViT+Transformer)', '').replace('  ', ' ').strip()
        s = s.replace('Cmp-1 ', '').replace('Cmp-3 ', '').replace('CNN-BiLSTM', 'EMSPN').replace('CNN Only', 'CNN')
        s = s.replace('(EMSPN)', '').strip()
        s = s.replace('RandomForest', 'RF')
        s = s.replace('ML ', '')
        return s

    def _style_for(model_name: str):
        upper = model_name.upper().replace('\n', ' ')
        if 'FULL MODEL' in upper or 'VIT+TRANSFORMER' in upper:
            return style_map[0]
        if 'CNN-BILSTM' in upper or 'EMSPN' in upper:
            return style_map[1]
        if 'CNN ONLY' in upper:
            return style_map[2]
        if 'MLP' in upper:
            return style_map[3]
        if 'SVM' in upper:
            return style_map[4]
        if 'RANDOMFOREST' in upper or 'RANDOM FOREST' in upper:
            return style_map[5]
        if 'XGBOOST' in upper:
            return style_map[6]
        return ('default', '#7f8c8d', 1.6, '-')

    def _apply_boxed_panel(ax, linewidth: float = 0.6) -> None:
        """统一为太古代时间演化图的黑色细边框风格。"""
        # 中文注释：ROC/PR 子图边框与太古代时间演化主图保持一致。
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#000000')
            spine.set_linewidth(linewidth)
            spine.set_capstyle('butt')
        ax.tick_params(width=linewidth)

    # ── 设置 SCI 期刊字体风格 ──────────────────────────────────
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.edgecolor':     '#000000',
        'axes.linewidth':    0.6,
        'axes.labelsize':    15,
        'axes.titlesize':    15,
        'xtick.labelsize':   13,
        'ytick.labelsize':   13,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'legend.fontsize':   12,
        'legend.frameon':    True,
        'legend.framealpha': 0.95,
        'legend.edgecolor':  '#000000',
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 计算所有模型的曲线及面积
    plot_records = []
    for r in ordered:
        y_true  = r['y_true']
        y_probs = r['y_probs']
        n_cls   = y_probs.shape[1]
        y_onehot = np.eye(n_cls)[y_true]

        # micro-averaged ROC；PR 曲线形状也用 micro，但 mAP 标注值取实验记录的宏平均
        fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_probs.ravel())
        try:
            roc_auc = roc_auc_score(y_onehot, y_probs, average='micro')
        except ValueError:
            roc_auc = auc(fpr, tpr)
        prec, rec, _ = _pr_curve(y_onehot.ravel(), y_probs.ravel())
        map_val = r.get('mAP', _ap(y_onehot, y_probs, average='micro'))

        color, lw, ls = _style_for(r['model_name'])[1:]
        plot_records.append({
            'name':   _pretty_label(r['model_name']),
            'fpr':    fpr, 'tpr': tpr, 'auc':  roc_auc,
            'prec':   prec, 'rec': rec, 'ap':   map_val,
            'color':  color, 'lw':  lw,  'ls':  ls,
        })

    # ── (a) ROC ─────────────────────────────────────────────────
    ax = axes[0]
    for rec in plot_records:
        ax.plot(rec['fpr'], rec['tpr'],
                color=rec['color'], lw=rec['lw'], linestyle=rec['ls'],
                label=f"{rec['name']} (AUC = {rec['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color='lightgray', lw=1.0, linestyle='--', zorder=0)

    ax.set_xlim(-0.005, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _apply_boxed_panel(ax)
    leg_a = ax.legend(loc='lower right', edgecolor='#000000',
                      fancybox=False, framealpha=0.95)
    leg_a.get_frame().set_linewidth(0.6)
    ax.text(-0.13, 1.04, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

    # ── (b) PR ──────────────────────────────────────────────────
    ax = axes[1]
    for rec in plot_records:
        ax.plot(rec['rec'], rec['prec'],
                color=rec['color'], lw=rec['lw'], linestyle=rec['ls'],
                label=f"{rec['name']} (mAP = {rec['ap']:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _apply_boxed_panel(ax)
    leg_b = ax.legend(loc='lower left', edgecolor='#000000',
                      fancybox=False, framealpha=0.95)
    leg_b.get_frame().set_linewidth(0.6)
    ax.text(-0.13, 1.04, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    save_path_png = os.path.join(output_dir, 'roc_pr_sci_comparison.png')
    plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
    plt.close()

    # 还原 rcParams（避免影响后续绘图）
    plt.rcdefaults()
    print(f'\nSCI 风格 ROC/PR 对比图已保存:')
    print(f'  PNG (600 dpi): {save_path_png}')

    # 同时保存对比指标到 CSV，方便论文表格引用
    summary_path = os.path.join(output_dir, 'roc_pr_sci_comparison_metrics.csv')
    pd.DataFrame([{
        'Model':        rec['name'],
        'AUC (micro)':  f"{rec['auc']:.4f}",
        'mAP (micro)':  f"{rec['ap']:.4f}",
    } for rec in plot_records]).to_csv(
        summary_path, index=False, encoding='utf-8-sig')
    print(f'  对比指标 CSV:  {summary_path}')


def save_per_seed_csv(all_dl_results, output_dir):
    """各深度学习实验所有种子的逐条记录写入 ablation_per_seed.csv。"""
    rows = []
    for r in all_dl_results:
        for sr in r.get('per_seed_results', [r]):
            rows.append({
                'Experiment': r['model_name'].replace('\n', ' '),
                'Seed':       sr.get('seed', 'N/A'),
                'BestEpoch':  sr.get('best_acc_epoch', 'N/A'),
                'Accuracy':   f"{sr['accuracy']:.4f}",
                'Precision':  f"{sr['precision']:.4f}",
                'Recall':     f"{sr['recall']:.4f}",
                'F1-Score':   f"{sr['f1_score']:.4f}",
                'Macro F1':   f"{sr['macro_f1']:.4f}",
                'mAP':        f"{sr['mAP']:.4f}",
                'Val Loss':   f"{sr['val_loss']:.4f}",
            })
    path = os.path.join(output_dir, 'ablation_per_seed.csv')
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f'\nPer-seed 明细表已保存: {path}')


# =============================================================
# 主函数
# =============================================================

if __name__ == '__main__':
    # 中文注释：路径按完整字符串写出，避免路径拼接导致运行环境差异。
    # 中文注释：读取SMOTE后的训练集；模型仍为原始单通道结构，不使用缺失掩码。
    # 中文注释：使用原始空间全局插补、无水标准化和分位数编码后的主流程数据。
    TRAIN_FILE = str(TRAIN_NORM_CSV)
    TEST_FILE = str(TEST_NORM_CSV)
    TRAIN_MASK_FILE = str(MASK_TRAIN_CSV)
    TEST_MASK_FILE = str(MASK_TEST_CSV)

    # 中文注释：正文最终方案固定为显式缺失编码，并保留原始6920条CFB训练样品。
    USE_MISSING_MASK = True
    CFB_TARGET_COUNT = None
    output_dir = str(MODELS_DIR)

    # 中文注释：训练轮数和随机种子直接在代码中配置，不再读取环境变量。
    EPOCHS = 200
    SEEDS = [42, 123]
    MIXUP_ALPHA      = 0
    # 中文注释：本轮只运行“原始空间全局插补 + CR及四个少数类SMOTE到3000 + 普通交叉熵”。
    RUN_ML_BASELINES = False

    # ════════════════════════════════════════════
    # 列排列方案选择
    #   'v1' : 矩阵=元素周期表顺序 + 序列=电极电势序列（原始方案，默认）
    #   'v2' : 矩阵=地化亲缘分组   + 序列=不相容性从高到低（历史消融方案）
    # ════════════════════════════════════════════
    COLUMN_ORDER_SCHEME = 'v1'

    if COLUMN_ORDER_SCHEME == 'v2':
        columns_to_extract = IMAGE_COLUMNS_V2
        seq_columns        = SEQUENCE_COLUMNS_V2
    else:
        columns_to_extract = ORIGINAL_IMAGE_COLUMNS
        seq_columns        = COLUMNS_ELECTRODE_ORDER_V1

    # ════════════════════════════════════════════
    # 实验选择
    # ════════════════════════════════════════════
    # EXPERIMENTS_TO_RUN = [
    #     'Full',    # 新主模型: ViT-Transformer 双流（无 CNN）
    #     'Abl-1',   # 消融【新增】: 仅 ViT 矩阵分支
    #     'Abl-2',   # 消融: 仅 Transformer 序列分支
    #     'Abl-3',   # 消融: 双流 w/o Positional Encoding
    #     'Cmp-1',   # 对比: CNN-BiLSTM (EMSPN 前作)
    #     'Cmp-2',   # 对比: CNN-ViT-Transformer (旧 Full，证 CNN 冗余)
    #     'Cmp-3',   # 对比: CNN Only
    # ]
    EXPERIMENTS_TO_RUN = [
        'Full'    # 新主模型: ViT-Transformer 双流（无 CNN）
    ]

    os.makedirs(output_dir, exist_ok=True)

    print(f'\n加载数据... (列排列方案: {COLUMN_ORDER_SCHEME})')
    print(f'GeoDAN显式缺失编码: {"开启" if USE_MISSING_MASK else "关闭"}')
    (X_train_img, X_train_seq, y_train,
     X_test_img,  X_test_seq,  y_test, unique_labels) = load_presplit_csv(
        TRAIN_FILE,
        TEST_FILE,
        columns_to_extract,
        seq_columns,
        use_missing_mask=USE_MISSING_MASK,
        train_mask_path=TRAIN_MASK_FILE,
        test_mask_path=TEST_MASK_FILE,
        cfb_target_count=CFB_TARGET_COUNT,
    )
    num_classes = len(unique_labels)

    ALL_EXPERIMENTS = {
        'Full':  ('Full Model\n(ViT+Transformer)',
                  lambda: ViT_Transformer_DualStream(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
        'Abl-1': ('Abl-1\nViT Only (Matrix)',
                  lambda: Ablation_ViT_Only(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
        'Abl-2': ('Abl-2\nTransformer Only (Seq)',
                  lambda: Ablation_Transformer_Only(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
        'Abl-3': ('Abl-3\nw/o Pos Encoding',
                  lambda: Ablation_NoPositionalEncoding(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
        'Cmp-1': ('Cmp-1\nCNN-BiLSTM (EMSPN)',
                  lambda: CNN_BiLSTM(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
        'Cmp-2': ('Cmp-2\nCNN-ViT-Transformer',
                  lambda: CNN_ViT_Transformer(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
        'Cmp-3': ('Cmp-3\nCNN Only',
                  lambda: Baseline_CNN_Only(
                      num_classes=num_classes,
                      use_missing_mask=USE_MISSING_MASK,
                  )),
    }
    # FULL_CONFIG = dict(embed_dim=64, num_heads=4,
    #                    transformer_layers=2, ff_dim=128, dropout=0.1)
    # ABL_SINGLE_CONFIG = dict(embed_dim=32, num_heads=4,
    #                          transformer_layers=1, ff_dim=64, dropout=0.2)

    # ALL_EXPERIMENTS = {
    #     'Full':  ('Full Model\n(ViT+Transformer)',
    #               lambda: ViT_Transformer_DualStream(
    #                   num_classes=num_classes, **FULL_CONFIG)),
    #     'Abl-1': ('Abl-1\nViT Only (Matrix)',
    #               lambda: Ablation_ViT_Only(
    #                   num_classes=num_classes, **ABL_SINGLE_CONFIG)),
    #     'Abl-2': ('Abl-2\nTransformer Only (Seq)',
    #               lambda: Ablation_Transformer_Only(
    #                   num_classes=num_classes, **ABL_SINGLE_CONFIG)),
    #     'Abl-3': ('Abl-3\nw/o Pos Encoding',
    #               lambda: Ablation_NoPositionalEncoding(
    #                   num_classes=num_classes, **FULL_CONFIG)),
    #     'Cmp-1': ('Cmp-1\nCNN-BiLSTM (EMSPN)',
    #               lambda: CNN_BiLSTM(num_classes=num_classes)),
    #     'Cmp-2': ('Cmp-2\nCNN-ViT-Transformer',
    #               lambda: CNN_ViT_Transformer(
    #                   num_classes=num_classes, **FULL_CONFIG)),
    #     'Cmp-3': ('Cmp-3\nCNN Only',
    #               lambda: Baseline_CNN_Only(num_classes=num_classes)),
    # }

    invalid_keys = [k for k in EXPERIMENTS_TO_RUN if k not in ALL_EXPERIMENTS]
    if invalid_keys:
        raise ValueError(
            f'EXPERIMENTS_TO_RUN 中存在未注册的实验编号: {invalid_keys}\n'
            f'可用编号: {list(ALL_EXPERIMENTS.keys())}')

    experiments = [ALL_EXPERIMENTS[k] for k in EXPERIMENTS_TO_RUN]

    print(f'\n本次将运行以下 {len(experiments)} 个实验: {EXPERIMENTS_TO_RUN}')
    print(f'每个实验重复 {len(SEEDS)} 个种子: {SEEDS}')
    print(f'打印频率: 每 {PRINT_EVERY} 个 epoch 一次')

    all_results = []
    total_start = time.time()

    for exp_name, model_factory in experiments:
        result = run_experiment_multi_seed(
            model_factory, exp_name,
            X_train_img, X_train_seq, y_train,
            X_test_img,  X_test_seq,  y_test,
            num_classes, device,
            epochs=EPOCHS, mixup_alpha=MIXUP_ALPHA,
            seeds=SEEDS,
            output_dir=output_dir,
            # 中文注释：仅保留最终最优种子权重，避免生成多个pth文件。
            save_per_seed_weights=False,
        )
        all_results.append(result)

        fname     = safe_filename(exp_name)
        save_path = os.path.join(output_dir, f'{fname}_best_seed.pth')
        torch.save(result['best_state_dict'], save_path)
        print(f'  最优种子权重已保存: {save_path}')

    save_per_seed_csv(all_results, output_dir)

    if RUN_ML_BASELINES:
        ml_results = run_ml_baseline_experiments(
            X_train_img, y_train,
            X_test_img, y_test,
            num_classes, unique_labels, output_dir,
            seeds=SEEDS,
        )
        all_results.extend(ml_results)

    total_elapsed = time.time() - total_start
    th, tm, ts = (int(total_elapsed // 3600),
                  int((total_elapsed % 3600) // 60),
                  int(total_elapsed % 60))
    print(f'\n全部实验总耗时: {th:02d}h {tm:02d}m {ts:02d}s')

    summary_rows = []
    for r in all_results:
        h_m_s = (int(r['elapsed_sec'] // 3600),
                 int((r['elapsed_sec'] % 3600) // 60),
                 int(r['elapsed_sec'] % 60))
        summary_rows.append({
            'Experiment':    r['model_name'].replace('\n', ' '),
            'Params':        r['num_params'],
            'Seeds':         str(r.get('seeds', [42])),
            'BestSeed':      r.get('best_seed', 'N/A'),
            'BestEpoch':     r.get('best_acc_epoch', 'N/A'),
            'Accuracy':      f"{r['accuracy']:.4f}",
            'Precision':     f"{r['precision']:.4f}",
            'Recall':        f"{r['recall']:.4f}",
            'F1-Score':      f"{r['f1_score']:.4f}",
            'Macro F1':      f"{r['macro_f1']:.4f}",
            'mAP':           f"{r['mAP']:.4f}",
            'Val Loss':      f"{r['val_loss']:.4f}",
            'Acc Std':       f"{r.get('accuracy_std', 0):.4f}",
            'F1 Std':        f"{r.get('f1_score_std', 0):.4f}",
            'Macro F1 Std':  f"{r.get('macro_f1_std', 0):.4f}",
            'mAP Std':       f"{r.get('mAP_std', 0):.4f}",
            'Time':          f"{h_m_s[0]:02d}h{h_m_s[1]:02d}m{h_m_s[2]:02d}s",
        })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 80)
    print('消融实验汇总表（多种子均值 ± 标准差）')
    print('=' * 80)
    display_cols = [c for c in summary_df.columns if c not in ('Acc Std', 'F1 Std', 'Macro F1 Std', 'mAP Std')]
    print(summary_df[display_cols].to_string(index=False))
    print(f'\n汇总表已保存: {summary_path}')

    plot_ablation_comparison(all_results, output_dir, unique_labels, _ml_label_mapping())
    plot_all_training_curves(all_results, output_dir)
    plot_individual_training_curves(all_results, output_dir)

    # ── SCI 风格 ROC + PR 对比图（主模型 vs CNN-BiLSTM vs 4 个 ML 基线） ──
    plot_roc_pr_sci_comparison(all_results, output_dir)

    print('\n所有消融实验完成！')
