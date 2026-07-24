# -*- coding: utf-8 -*-
"""
SHAP 汇总图 —— 直接运行 SHAP，并把各子图分别单独成图
==================================================
与之前不同：本脚本 **直接计算 SHAP**（不再读取 shap_merged.npy），
完全复用 ``shap_vit_transformer_dualstream.py`` 的模型结构、数据加载、
背景采样与 SHAP 计算逻辑；新增 **样本数控制**，默认给一个较小值方便测试。

输出（每个子图单独成图，便于在 Visio 自由拼接）：
  Figure7a_heatmap.png           —— (a) 9 类 × 36 元素 median(|SHAP|) 热图
  Figure7b_direction.png         —— (b) Top-15 元素 SHAP 方向热图（蓝-白-红）
  Figure7c_ranking.png           —— (c) 真实类别输出 Top-20 稳健重要性排名
  Figure7d_beeswarm_<setting>.*  —— (d) 各构造环境真实类别样本 Top-8 蜂窝图

样本数控制（脚本默认变量）
------------------------
默认：N_EXPLAIN_PER_CLASS = 300，N_BACKGROUND = 1000，BATCH_SIZE = 50。
背景样本按训练集类别比例分层抽样；待解释样本按类别配额抽样。
构造环境子图默认只统计该真实类别样本，并用 median(|SHAP|) 做稳健排名。
如需调整样本数、输出目录或是否绘制蜂窝图，直接修改下方默认配置变量即可。

运行环境
--------
需同时具备 torch 与 shap（本机为 conda ``torch`` 环境，已安装 shap）。
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 真正复用原脚本的模型 / 数据 / SHAP 计算（torch + shap 真实导入）──
import shap_vit_transformer_dualstream as base
from shap_vit_transformer_dualstream import (
    ALL_FEATURES_COLS,
    COL_DISPLAY,
    LABEL_MAPPING,
    OUTPUT_DIR,
    TRAIN_FILE,
    TEST_FILE,
    TRAIN_MASK_FILE,
    TEST_MASK_FILE,
    USE_MISSING_MASK,
    MODEL_PATH,
    COLUMNS_IMG_ORDER,
    COLUMNS_ELECTRODE_ORDER,
    load_data,
    select_background,
    compute_shap_values,
    ViT_Transformer_DualStream,
    TableInputWrapper,
)
import torch

# 中文注释：SHAP 面板输出目录使用当前项目的集中路径配置。
from config.paths import SHAP_FIGURE7_PANEL_DIR


# ══════════════════════════════════════════════════════════════
# ①  默认运行配置（取消 argparse，直接修改这里的变量即可）
# ══════════════════════════════════════════════════════════════
N_EXPLAIN_PER_CLASS = 300   # 未单独指定类别的抽样上限；样本不足时全取
# 中文注释：只降低四个样本量较大/对 Ba 较敏感类别的解释样本数，不按 Ba 数值筛样本。
EXPLAIN_CLASS_LIMITS = {
    'CF': 200,
    'CR': 200,
    'OI': 200,
    'MOR': 200,
}
EXPLAIN_FEATURE_QUANTILE_FILTERS = {}
N_BACKGROUND = 1000
BATCH_SIZE = 50
RANDOM_SEED = 42
MODEL_WEIGHT_PATH = MODEL_PATH
OUTPUT_PANEL_DIR = str(SHAP_FIGURE7_PANEL_DIR)
DRAW_BEESWARM = True

# 中文注释：如果缓存来自不同抽样策略，不要复用；正式无 Ba 区间筛选时建议重新计算。
USE_CACHED_SHAP = False
CACHED_SHAP_N = 2023

# 中文注释：Figure7c 排名口径。
# mean_all_outputs: 传统 mean(|SHAP|)，会强化跨类别输出和少数强响应样本；
# true_class_median: 只取样本真实类别对应输出，并用中位数做稳健排名。
RANKING_IMPORTANCE_STAT = 'true_class_median'

# ── 蜂窝图(d)密度控制：每行随机抽样上限 + 点样式（只影响绘图，不改 SHAP 计算）──
BEESWARM_MAX_POINTS = 180   # 每个元素行最多画多少点；None 表示全画
BEESWARM_POINT_SIZE = 7     # 散点大小（原为 13）
BEESWARM_ALPHA = 0.65       # 不透明度（原为 0.92）


# ══════════════════════════════════════════════════════════════
# ②  地球化学元素分组（仅供地化解释，不代表贡献排序）
# ══════════════════════════════════════════════════════════════
ELEMENT_GROUPS = [
    ('LILE',       ['Rb', 'Ba', 'Sr', 'K']),
    ('HFSE',       ['Th', 'Nb', 'Ta', 'Zr', 'Hf']),
    ('REE',        ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
                    'Tb', 'Dy', 'Ho', 'Er', 'Yb', 'Lu', 'Y']),
    ('Transition', ['V', 'Cr', 'Mn', 'Co', 'Ni']),
    ('Major',      ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'P']),
]
# 中文注释：Figure7a 仅调整横向展示顺序，让 Major elements 紧跟 LILE；不改变任何 SHAP 统计。
PANEL_A_ELEMENT_GROUPS = [
    ('LILE',       ['Rb', 'Ba', 'Sr', 'K']),
    ('Major',      ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'P']),
    ('HFSE',       ['Th', 'Nb', 'Ta', 'Zr', 'Hf']),
    ('REE',        ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
                    'Tb', 'Dy', 'Ho', 'Er', 'Yb', 'Lu', 'Y']),
    ('Transition', ['V', 'Cr', 'Mn', 'Co', 'Ni']),
]
GROUP_COLORS = {
    'LILE':       '#D98C00',   # 柔和橙
    'HFSE':       '#7A3E9D',   # 柔和紫
    'REE':        '#5AA05A',   # 柔和绿
    'Transition': '#56B7B1',   # 柔和青绿
    'Major':      '#5B84B1',   # 柔和蓝
}
GROUP_FULLNAME = {
    'LILE': 'LILE', 'HFSE': 'HFSE', 'REE': 'REE',
    'Transition': 'Transition metals', 'Major': 'Major elements',
}
GROUP_ORDER = [g for g, _ in ELEMENT_GROUPS]
SYM2GROUP = {sym: g for g, syms in ELEMENT_GROUPS for sym in syms}
GROUP_RANK = {g: i for i, g in enumerate(GROUP_ORDER)}

DIRECTION_CMAP = LinearSegmentedColormap.from_list(
    'blue_to_red', ['#2166AC', '#B2182B'])

TECTONIC_ORDER = ['BAB', 'CA', 'CF', 'CR', 'IOA', 'IA', 'OI', 'OP', 'MOR']
# 中文注释：Figure7a 专用 y 轴展示顺序；只影响热图行顺序，不改变 SHAP 统计。
PANEL_A_TECTONIC_ORDER = ['CA', 'IA', 'IOA', 'BAB', 'MOR', 'OP', 'OI', 'CR', 'CF']
TECTONIC_FULLNAME = {
    'BAB': 'Back-arc Basin',          'CA': 'Continental Arc',
    'CF': 'Continental Flood Basalt', 'CR': 'Continental Rift',
    'IOA': 'Intra-oceanic Arc',       'IA': 'Island Arc',
    'OI': 'Ocean Island',             'OP': 'Ocean Plateau',
    'MOR': 'Mid-Ocean Ridge',
}


def _set_pub_rcparams():
    rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         9,
        'axes.linewidth':    0.7,
        'axes.edgecolor':    '#333333',
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size':  2.8,
        'ytick.major.size':  2.8,
        'xtick.color':       '#333333',
        'ytick.color':       '#333333',
        'axes.labelcolor':   '#111111',
        'axes.titlecolor':   '#111111',
        'figure.facecolor':  'white',
        'savefig.facecolor': 'white',
        'svg.fonttype':      'none',
        'ps.fonttype':       42,
    })


def _save(fig, output_dir, name):
    png = os.path.join(output_dir, name + '.png')
    fig.savefig(png, dpi=1200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    mpl.rcParams.update(mpl.rcParamsDefault)
    print(f'  已保存：{png}')


def _ordered_classes(unique_labels, tectonic_order=None):
    """按 TECTONIC_ORDER 返回 (short_codes, idx_into_merged_shap)。"""
    if tectonic_order is None:
        tectonic_order = TECTONIC_ORDER
    short_of = [LABEL_MAPPING.get(lbl, lbl) for lbl in unique_labels]
    short2pos = {s: i for i, s in enumerate(short_of)}
    ordered, idx = [], []
    for code in tectonic_order:
        if code in short2pos:
            ordered.append(code)
            idx.append(short2pos[code])
    for s, p in short2pos.items():
        if s not in ordered:
            ordered.append(s)
            idx.append(p)
    return ordered, idx


def _draw_group_bars(ax, col_groups, y_line, y_text, lw=3.4, fontsize=8,
                     text_color=None, fontweight='normal'):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    n = len(col_groups)
    start = 0
    while start < n:
        g = col_groups[start]
        end = start
        while end + 1 < n and col_groups[end + 1] == g:
            end += 1
        ax.plot([start - 0.42, end + 0.42], [y_line, y_line], transform=trans,
                color=GROUP_COLORS[g], lw=lw, solid_capstyle='butt',
                clip_on=False, zorder=5)
        ax.text((start + end) / 2.0, y_text, GROUP_FULLNAME[g], transform=trans,
                ha='center', va='top', fontsize=fontsize,
                color=text_color or GROUP_COLORS[g],
                fontweight=fontweight, clip_on=False)
        start = end + 1


def _bold_ticklabels(ax):
    """加粗坐标轴刻度文字。"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


# ══════════════════════════════════════════════════════════════
# ③  计算 SHAP（带样本数控制）
# ══════════════════════════════════════════════════════════════
def _resolve_model_path(model_path):
    """优先用传入/默认路径；不存在则回退到 GeoDAN 目录下的 Full_Model 权重。"""
    if model_path and os.path.exists(model_path):
        return model_path
    fallback = os.path.join(
        os.path.dirname(OUTPUT_DIR), 'Full_Model_(ViT+Transformer)_best_seed.pth')
    if os.path.exists(fallback):
        print(f'[提示] 未找到 {model_path}，改用默认权重：{fallback}')
        return fallback
    raise FileNotFoundError(
        f'模型权重不存在：{model_path}；回退路径也不存在：{fallback}。'
        f'请用 --model 指定正确的 .pth。')


def _select_explain_samples_per_class(
        X_test_36, y_test, unique_labels, n_per_class, class_limits=None,
        feature_quantile_filters=None):
    """按类别配额从测试集抽样，保证各构造环境参与 SHAP 的样本数更均衡。"""
    class_limits = class_limits or {}
    feature_quantile_filters = feature_quantile_filters or {}
    explain_idx = []
    for lbl in np.unique(y_test):
        lbl_idx = np.where(y_test == lbl)[0]
        full_label = unique_labels[int(lbl)]
        short_label = LABEL_MAPPING.get(full_label, full_label)
        class_limit = class_limits.get(short_label, n_per_class)

        candidate_idx = lbl_idx
        filter_config = feature_quantile_filters.get(short_label)
        filter_message = ''
        if filter_config is not None:
            feature_name, lower_q, upper_q = filter_config
            feature_idx = ALL_FEATURES_COLS.index(feature_name)
            class_values = X_test_36[lbl_idx, feature_idx]
            lower_value, upper_value = np.quantile(
                class_values, [lower_q, upper_q])
            keep = (
                (class_values >= lower_value)
                & (class_values <= upper_value)
            )
            candidate_idx = lbl_idx[keep]
            filter_message = (
                f'，{feature_name} 保留 P{lower_q * 100:.0f}'
                f'--P{upper_q * 100:.0f}，候选 {len(candidate_idx)}'
            )

        n_lbl = min(class_limit, len(candidate_idx))
        picked_idx = np.random.choice(candidate_idx, n_lbl, replace=False)
        explain_idx.extend(picked_idx)

        print(f'  {short_label}: 抽取 {n_lbl} / {len(lbl_idx)} 个测试样本'
              f'（上限 {class_limit}{filter_message}）')

    return X_test_36[explain_idx], explain_idx


def compute_merged_shap(
        n_explain_per_class, n_background, batch_size, model_path,
        class_limits=None, feature_quantile_filters=None):
    # 中文注释：与正式 GeoDAN 一致开启缺失编码，load_data 返回 72 维 [数值|mask]。
    X_train, y_train, X_test, y_test, unique_labels = load_data(
        TRAIN_FILE, TEST_FILE, TRAIN_MASK_FILE, TEST_MASK_FILE)
    num_classes = len(unique_labels)

    model_path = _resolve_model_path(model_path)
    model = ViT_Transformer_DualStream(
        num_classes=num_classes, use_missing_mask=USE_MISSING_MASK).to(base.device)
    state_dict = torch.load(model_path, map_location=base.device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'模型权重加载成功：{model_path}')

    wrapped = TableInputWrapper(
        model, canonical_cols=ALL_FEATURES_COLS,
        img_cols=COLUMNS_IMG_ORDER, seq_cols=COLUMNS_ELECTRODE_ORDER,
        use_missing_mask=USE_MISSING_MASK,
    ).to(base.device)
    wrapped.eval()

    n_background = min(n_background, len(y_train))
    np.random.seed(RANDOM_SEED)
    print(f'分层采样背景数据 {n_background} 个 ...')
    background_tensor = select_background(X_train, y_train, n_background)

    # 中文注释：按类别配额抽样，降低大类主导全局 SHAP 汇总的程度。
    np.random.seed(RANDOM_SEED)
    print(f'按类别配额抽取待解释测试样本，默认上限 {n_explain_per_class} 个 ...')
    X_exp, explain_idx = _select_explain_samples_per_class(
        X_test, y_test, unique_labels, n_explain_per_class, class_limits,
        feature_quantile_filters)
    print(f'待解释样本：{len(explain_idx)} 个（来自测试集）')

    shap_values = compute_shap_values(
        wrapped, background_tensor, X_exp,
        n_explain=len(explain_idx), batch_size=batch_size)
    merged_full = np.array(shap_values)            # (n_classes, n_samples, F)

    # 中文注释：开启缺失编码时 F=72=[数值|mask]，绘图只取数值通道的 36 维归因；
    #           X_exp 同步切到数值通道，供 (b)/(d) 按特征值上色。
    n_feat = len(ALL_FEATURES_COLS)
    if merged_full.shape[2] == 2 * n_feat:
        merged_shap = merged_full[:, :, :n_feat]
        X_exp_36 = X_exp[:, :n_feat]
    else:
        merged_shap = merged_full
        X_exp_36 = X_exp
    explain_idx = np.asarray(explain_idx)
    y_exp = y_test[explain_idx]
    return merged_shap, X_exp_36, unique_labels, explain_idx, y_exp


def load_cached_shap(output_dir, cache_n):
    """读取已有 SHAP 缓存，并用 explain_idx 还原每个解释样本的真实类别。"""
    shap_path = os.path.join(output_dir, f'shap_merged_n{cache_n}.npy')
    x_path = os.path.join(output_dir, f'X_exp_36_n{cache_n}.npy')
    idx_path = os.path.join(output_dir, f'explain_idx_n{cache_n}.npy')
    if not (os.path.exists(shap_path) and os.path.exists(x_path)
            and os.path.exists(idx_path)):
        raise FileNotFoundError(
            f'缓存不完整，请检查：{shap_path} / {x_path} / {idx_path}')

    merged_shap = np.load(shap_path)
    X_exp_36 = np.load(x_path)
    explain_idx = np.load(idx_path)
    # 中文注释：只为拿到 unique_labels 和测试集标签，不重新计算 SHAP。
    _, _, _, y_test, unique_labels = load_data(
        TRAIN_FILE, TEST_FILE, TRAIN_MASK_FILE, TEST_MASK_FILE)
    y_exp = y_test[explain_idx]
    print(f'已读取 SHAP 缓存：{shap_path}')
    return merged_shap, X_exp_36, unique_labels, explain_idx, y_exp


# ══════════════════════════════════════════════════════════════
# ④  (a) 单独：真实类别样本 median(|SHAP|) 热图
# ══════════════════════════════════════════════════════════════
def _class_sample_mask(y_exp, class_idx, n_samples):
    """返回某输出类别对应的真实样本掩码；无 y_exp 时退回全样本。"""
    if y_exp is None:
        return np.ones(n_samples, dtype=bool)
    return np.asarray(y_exp) == class_idx


def _class_feature_importance(merged_shap, class_idx, y_exp=None):
    """用该真实类别样本计算某输出类别的稳健特征重要性。"""
    mask = _class_sample_mask(y_exp, class_idx, merged_shap.shape[1])
    if not mask.any():
        return np.zeros(merged_shap.shape[2])
    return np.median(np.abs(merged_shap[class_idx, mask, :]), axis=0)


def plot_panel_a(merged_shap, unique_labels, output_dir, y_exp=None):
    _set_pub_rcparams()
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    sym2idx = {s: i for i, s in enumerate(display_names)}
    ordered_codes, cls_idx = _ordered_classes(unique_labels, PANEL_A_TECTONIC_ORDER)

    flat_syms = [s for _, syms in PANEL_A_ELEMENT_GROUPS
                 for s in syms if s in sym2idx]
    col_order = [sym2idx[s] for s in flat_syms]
    col_groups = [SYM2GROUP[s] for s in flat_syms]
    data = np.vstack([
        _class_feature_importance(merged_shap, c, y_exp)
        for c in cls_idx
    ])[:, col_order]

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=1200)
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(flat_syms)))
    ax.set_xticklabels(flat_syms, fontsize=8.5, rotation=0, fontweight='bold')
    ax.set_yticks(range(len(ordered_codes)))
    ax.set_yticklabels(ordered_codes, fontsize=10, fontweight='semibold')
    # 中文注释：缩小左侧类别标签和热图主体的距离，同时保持半粗体出版风格。
    ax.tick_params(axis='y', pad=0)
    ax.set_ylabel('Tectonic settings', fontsize=10, labelpad=2,
                  fontweight='semibold')
    ax.tick_params(axis='x', length=0)
    # ax.set_title('(a)  True-class SHAP importance heatmap (median |SHAP value|)',
    #              fontsize=12, loc='left', pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.018, shrink=0.85)
    cb.set_label('median(|SHAP|)', fontsize=10, labelpad=5,
                 fontweight='semibold')
    cb.ax.tick_params(labelsize=8, length=2)
    # 中文注释：色带刻度数字需要逐个设置字重，tick_params 不能控制 fontweight。
    for tick_label in cb.ax.get_yticklabels():
        tick_label.set_fontweight('semibold')
    cb.outline.set_linewidth(0.6)
    _draw_group_bars(ax, col_groups, y_line=-0.11, y_text=-0.20,
                     lw=3.6, fontsize=10, text_color='#111111',
                     fontweight='bold')
    fig.subplots_adjust(left=0.055, right=0.995, top=0.9, bottom=0.24)
    _save(fig, output_dir, 'Figure7a_heatmap')


# ══════════════════════════════════════════════════════════════
# ⑤  (b) 单独：Top-15 方向热图
# ══════════════════════════════════════════════════════════════
def plot_panel_b(merged_shap, X_exp_36, unique_labels, output_dir, y_exp=None):
    _set_pub_rcparams()
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    ordered_codes, cls_idx = _ordered_classes(unique_labels)
    global_imp, _ = _ranking_importance(merged_shap, y_exp)

    top15 = np.argsort(global_imp)[::-1][:15]
    top15 = sorted(top15, key=lambda j: (GROUP_RANK[SYM2GROUP[display_names[j]]],
                                         -global_imp[j]))
    top15_syms = [display_names[j] for j in top15]
    direction = np.zeros((len(ordered_codes), len(top15)))
    for jj, feat_j in enumerate(top15):
        for ci, c in enumerate(cls_idx):
            mask = _class_sample_mask(y_exp, c, merged_shap.shape[1])
            fv = X_exp_36[mask, feat_j]
            sv = merged_shap[c, mask, feat_j]
            if len(fv) == 0:
                continue
            q25, q75 = np.percentile(fv, 25), np.percentile(fv, 75)
            high, low = fv >= q75, fv <= q25
            hi = sv[high].mean() if high.any() else 0.0
            lo = sv[low].mean() if low.any() else 0.0
            direction[ci, jj] = hi - lo
    dmax = np.max(np.abs(direction))
    direction_n = direction / dmax if dmax > 0 else direction

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=1200)
    im = ax.imshow(direction_n, aspect='auto',
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(top15_syms)))
    ax.set_xticklabels(top15_syms, fontsize=10, rotation=0)
    ax.set_yticks(range(len(ordered_codes)))
    ax.set_yticklabels(ordered_codes, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', length=0)
    # ax.set_title('(b)  SHAP direction heatmap (top 15 elements)',
    #              fontsize=12, loc='left', pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('SHAP value (direction)', fontsize=9, labelpad=3)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.12)
    _save(fig, output_dir, 'Figure7b_direction')


# ══════════════════════════════════════════════════════════════
# ⑥  (c) 单独：真实类别输出 Top-20 稳健重要性排名（按分组着色）
# ══════════════════════════════════════════════════════════════
def _ranking_importance(merged_shap, y_exp=None):
    """返回 Figure7c 使用的重要性向量和横轴标签。"""
    abs_shap = np.abs(merged_shap)
    if RANKING_IMPORTANCE_STAT == 'true_class_median':
        if y_exp is None:
            raise ValueError('true_class_median 需要 y_exp 真实类别标签')
        true_class_abs = np.asarray([
            abs_shap[int(y_exp[i]), i, :]
            for i in range(len(y_exp))
        ])
        return np.median(true_class_abs, axis=0), 'median(|SHAP|)'
    if RANKING_IMPORTANCE_STAT == 'median_all_outputs':
        return np.median(abs_shap, axis=(0, 1)), 'median(|SHAP|)'
    return abs_shap.mean(axis=(0, 1)), 'mean(|SHAP|)'


def plot_panel_c(merged_shap, output_dir, y_exp=None):
    _set_pub_rcparams()
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    global_imp, x_label = _ranking_importance(merged_shap, y_exp)

    top20 = np.argsort(global_imp)[::-1][:20]
    syms = [display_names[j] for j in top20]
    vals = global_imp[top20]
    # 暖色单色系（与 (a) 黄-橙-红热图协调）：前 N 个用稍深橙红突出，其余柔和橙
    N_HIGHLIGHT = 5
    C_SOFT = '#FDBB84'   # 柔和橙（普通柱）
    C_DEEP = '#E34A33'   # 橙红（Top 元素突出）
    cols = [C_DEEP if i < N_HIGHLIGHT else C_SOFT for i in range(len(top20))]
    y = np.arange(len(top20))
    # x 轴右边界调紧，减少柱子右侧空白；倍率太小会裁掉最长柱。
    x_axis_max = vals.max() * 1.03

    # 中文注释：略微增加 Figure7c 高度，避免 Top-20 排名行之间过于拥挤。
    fig, ax = plt.subplots(figsize=(3.6, 3.5), dpi=1200)
    ax.barh(y, vals, color=cols, edgecolor='none',
            height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(syms, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=10, labelpad=3,
                  fontweight='semibold')
    ax.set_xlim(0, x_axis_max)
    # ax.set_title('(c)  True-class robust SHAP importance ranking', fontsize=12,
    #              loc='left', pad=8)
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='y', length=3.2, width=0.7, color='#333333')
    ax.tick_params(axis='x', length=3.2, width=0.7, color='#333333')
    _bold_ticklabels(ax)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#cccccc',
            alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    # 用矩形边框把整张图包围起来（四条边都显示），并恢复 y 轴刻度线
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color('#333333')
    handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor=C_DEEP,
               markeredgecolor='none', markersize=7,
               label=f'Top {N_HIGHLIGHT} elements'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=C_SOFT,
               markeredgecolor='none', markersize=7, label='Other elements'),
    ]
    # 图例位置微调：left 越大越靠右，top 越大越靠上。
    legend_left = 0.6
    legend_top = 0.36
    # legend = ax.legend(handles=handles, title='SHAP ranking', loc='upper left',
    #                    bbox_to_anchor=(legend_left, legend_top),
    #                    fontsize=7.5, title_fontsize=7.5, frameon=True, framealpha=0.94,
    #                    edgecolor='#999999', handlelength=0.9, labelspacing=0.55,
    #                    borderpad=0.35)
    # legend.get_title().set_fontweight('bold')
    # for text in legend.get_texts():
    #     text.set_fontweight('bold')
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.08)
    _save(fig, output_dir, 'Figure7c_ranking')


# ══════════════════════════════════════════════════════════════
# ⑦  (d) 各构造环境真实类别样本 Top-8 蜂窝图（单独保存）
# ══════════════════════════════════════════════════════════════
def _beeswarm_offsets(x_values, bin_count=60, step=0.032, max_width=0.40):
    """按 SHAP 值分箱生成上下交错偏移，画出更接近 SHAP 原生风格的蜂窝点。"""
    x_values = np.asarray(x_values)
    offsets = np.zeros(len(x_values), dtype=float)
    finite = np.isfinite(x_values)
    if not finite.any():
        return offsets

    x_min, x_max = np.percentile(x_values[finite], [1, 99])
    if x_max <= x_min:
        x_min, x_max = np.min(x_values[finite]), np.max(x_values[finite])
    if x_max <= x_min:
        return offsets

    bins = np.linspace(x_min, x_max, bin_count + 1)
    bin_ids = np.clip(np.digitize(x_values, bins) - 1, 0, bin_count - 1)
    for bin_id in np.unique(bin_ids):
        idx = np.where(bin_ids == bin_id)[0]
        idx = idx[np.argsort(x_values[idx])]
        for k, point_idx in enumerate(idx):
            if k == 0:
                offsets[point_idx] = 0.0
                continue
            level = (k + 1) // 2
            sign = 1 if k % 2 else -1
            offsets[point_idx] = sign * min(level * step, max_width)

    return offsets


def _scatter_beeswarm(shap_vals, feat_vals, feature_names, ax):
    """绘制 SHAP 风格散点蜂窝图，颜色按每个特征值在本特征内归一化。"""
    shap_vals = np.asarray(shap_vals)
    feat_vals = np.asarray(feat_vals)
    n_features = shap_vals.shape[1]
    y_positions = np.arange(n_features)
    cmap = LinearSegmentedColormap.from_list(
        'feature_value_strong', ['#1F5AA6', '#F0F0F0', '#D7191C'])

    for row in range(n_features):
        sv = shap_vals[:, row]
        fv = feat_vals[:, row]

        # 每行随机抽样，避免点过密糊成实色带（颜色映射用全量数据估计分位数）。
        finite = np.isfinite(fv)
        if finite.any():
            vmin, vmax = np.percentile(fv[finite], [2, 98])
        else:
            vmin, vmax = 0.0, 1.0
        if BEESWARM_MAX_POINTS is not None and len(sv) > BEESWARM_MAX_POINTS:
            sel = np.random.choice(len(sv), BEESWARM_MAX_POINTS, replace=False)
            sv, fv = sv[sel], fv[sel]

        if vmax > vmin:
            color_value = np.clip((fv - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            color_value = np.full_like(fv, 0.5, dtype=float)

        # 按横向位置分箱堆叠点，比随机抖动更接近经典 SHAP beeswarm。
        jitter = _beeswarm_offsets(sv)
        ax.scatter(
            sv, y_positions[row] + jitter,
            c=color_value, cmap=cmap, vmin=0, vmax=1,
            s=BEESWARM_POINT_SIZE, alpha=BEESWARM_ALPHA,
            edgecolors='none', linewidths=0,
            rasterized=True, zorder=3,
        )

    for y in y_positions:
        ax.axhline(y, color='#d9d9d9', linestyle=(0, (2, 3)),
                   linewidth=0.6, zorder=0)
    ax.axvline(0, color='#555555', linewidth=1.0, zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value', fontsize=12, labelpad=4)
    ax.tick_params(axis='x', labelsize=9)
    _bold_ticklabels(ax)
    ax.grid(axis='x', linestyle='--', linewidth=0.8,
            color='#d0d0d0', alpha=0.85, zorder=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def plot_beeswarm_top8(merged_shap, X_exp_36, unique_labels, output_dir, y_exp=None):
    """每类只用真实属于该类的样本；Top8 按 median(|SHAP|) 排序，散点保留原始 SHAP 分布。"""
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    feat_arr = np.asarray(X_exp_36)
    ordered_codes, cls_idx = _ordered_classes(unique_labels)
    for code, c in zip(ordered_codes, cls_idx):
        mask = _class_sample_mask(y_exp, c, merged_shap.shape[1])
        if not mask.any():
            print(f'  [跳过] {code}: 没有对应真实类别样本')
            continue
        imp = _class_feature_importance(merged_shap, c, y_exp)
        top8 = np.argsort(imp)[::-1][:8]
        sv8, fv8 = merged_shap[c][mask][:, top8], feat_arr[mask][:, top8]
        names8 = [display_names[j] for j in top8]

        np.random.seed(42)
        _set_pub_rcparams()
        fig, ax = plt.subplots(figsize=(3.8, 2.4), dpi=1200)
        _scatter_beeswarm(sv8, fv8, names8, ax)
        # ax.set_title(f'SHAP — {code} (top 8 elements)', fontsize=11, pad=8)
        fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.13)
        safe = code.replace(' ', '_').replace('/', '_')
        _save(fig, output_dir, f'Figure7d_beeswarm_{safe}')


# ══════════════════════════════════════════════════════════════
# ⑧  主程序
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_PANEL_DIR, exist_ok=True)
    print(f'输出目录：{OUTPUT_PANEL_DIR}')
    print(f'样本设置：默认类别上限={N_EXPLAIN_PER_CLASS}  '
          f'类别覆盖={EXPLAIN_CLASS_LIMITS}  n_background={N_BACKGROUND}  '
          f'batch_size={BATCH_SIZE}\n')
    print(f"特征分位筛选：{EXPLAIN_FEATURE_QUANTILE_FILTERS or '无'}\n")
    print(f'构造环境子图口径：真实类别样本 + median(|SHAP|)\n')

    if USE_CACHED_SHAP:
        merged_shap, X_exp_36, unique_labels, explain_idx, y_exp = load_cached_shap(
            OUTPUT_PANEL_DIR, CACHED_SHAP_N)
    else:
        merged_shap, X_exp_36, unique_labels, explain_idx, y_exp = compute_merged_shap(
            N_EXPLAIN_PER_CLASS, N_BACKGROUND, BATCH_SIZE, MODEL_WEIGHT_PATH,
            EXPLAIN_CLASS_LIMITS, EXPLAIN_FEATURE_QUANTILE_FILTERS)

        # 保存本次（小样本）结果，文件名带样本数，绝不覆盖原 shap_merged.npy
        np.save(os.path.join(OUTPUT_PANEL_DIR, f'shap_merged_n{merged_shap.shape[1]}.npy'),
                merged_shap)
        np.save(os.path.join(OUTPUT_PANEL_DIR, f'X_exp_36_n{X_exp_36.shape[0]}.npy'), X_exp_36)
        # 中文注释：保存原测试集行号，确保条件抽样结果可以逐样本追溯。
        np.save(os.path.join(OUTPUT_PANEL_DIR, f'explain_idx_n{len(explain_idx)}.npy'),
                explain_idx)

    print('\n绘制 (a) 热图 ...')
    plot_panel_a(merged_shap, unique_labels, OUTPUT_PANEL_DIR, y_exp)
    print('绘制 (b) 方向热图 ...')
    plot_panel_b(merged_shap, X_exp_36, unique_labels, OUTPUT_PANEL_DIR, y_exp)
    print('绘制 (c) 重要性排名 ...')
    plot_panel_c(merged_shap, OUTPUT_PANEL_DIR, y_exp)
    if DRAW_BEESWARM:
        print('绘制 (d) 各构造环境 Top-8 蜂窝图 ...')
        plot_beeswarm_top8(merged_shap, X_exp_36, unique_labels,
                           OUTPUT_PANEL_DIR, y_exp)

    print(f'\n完成。所有子图已分别保存至：{OUTPUT_PANEL_DIR}')


if __name__ == '__main__':
    main()
