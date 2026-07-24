"""
==========================================================================
玄武岩地球化学缺失值插补 —— 方法对比脚本（附录用）
==========================================================================
对比 6 种插补方法在现代玄武岩数据集上的精度差异：
  - Mean        : 均值插补（baseline 1）
  - Median      : 中位数插补（baseline 2）
  - KNN         : K 近邻 (距离加权)
  - MICE        : 链式方程多重插补 (BayesianRidge 估计器)
  - MissForest  : 基于随机森林的非参数插补（正文使用方法）

评价方式（关键）：
  1. 按构造环境 CSV 分别做 5 折交叉验证
  2. 在测试集上按 MCAR 假设人为掩码 10% 观测值
     (只掩码原本有值的位置，避免对 NaN 评价无 ground truth 的问题)
  3. 每一折只用训练折 fit StandardScaler 和插补器，预测测试折
  4. **仅在被人为掩码的位置上**计算 MAE / RMSE / R²
     (在未掩码位置上插补器原样保留观测值，残差恒为 0，会稀释指标)
  5. 总体指标基于所有构造环境、所有折、所有掩码点合并后重新计算

输出：
  - results_by_fold.csv          每类构造环境、每折、逐方法结果
  - results_by_setting.csv       每类构造环境下各方法平均结果
  - results_overall.csv          所有构造环境掩码点合并后的总体结果
  - fig_imputation_comparison.png  主对比图，含 3 个子图

本脚本不保存任何插补模型。
==========================================================================
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 中文注释：插补对比附图读取当前项目现代合并总表，并输出到 05_imputed/imputation_comparison_output。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import COMBINED_CSV, ZENODO_MODERN_CSV, IMPUTATION_COMPARISON_DIR


# ============================================================================
# 配置区（直接修改这里）
# ============================================================================

# 中文注释：当前流程不再依赖旧的按构造环境 clean CSV，直接从现代合并总表按标签动态分组。
FULL_DATA_CSV = str(ZENODO_MODERN_CSV if Path(ZENODO_MODERN_CSV).exists() else COMBINED_CSV)
LABEL_COLUMN = "TECTONIC SETTING"

# 中文注释：输出统一写入当前项目的插补对比附图目录。
OUTPUT_DIR = str(IMPUTATION_COMPARISON_DIR)
RESULTS_BY_FOLD_CSV = str(IMPUTATION_COMPARISON_DIR / "results_by_fold.csv")
RESULTS_BY_SETTING_CSV = str(IMPUTATION_COMPARISON_DIR / "results_by_setting.csv")
RESULTS_OVERALL_CSV = str(IMPUTATION_COMPARISON_DIR / "results_overall.csv")
MAIN_FIGURE_PNG = str(IMPUTATION_COMPARISON_DIR / "fig_imputation_comparison.png")
ELEMENT_MISSING_RATIO_CSV = str(IMPUTATION_COMPARISON_DIR / "element_missing_ratio.csv")
ELEMENT_MISSING_RATIO_PNG = str(IMPUTATION_COMPARISON_DIR / "fig_element_missing_ratio.png")
# 评价参数
N_FOLDS = 5
MASK_RATIO = 0.10          # 人为掩码比例：降到 10% 后重新评估
RANDOM_SEED = 42

# 36 个化学元素
CHEMICAL_COLUMNS = [
    'NA2O(WT%)', 'MGO(WT%)', 'AL2O3(WT%)', 'SIO2(WT%)', 'P2O5(WT%)',
    'K2O(WT%)', 'CAO(WT%)', 'TIO2(WT%)', 'MNO(WT%)', 'FEOT(WT%)',
    'RB(PPM)', 'V(PPM)', 'CR(PPM)', 'CO(PPM)', 'NI(PPM)', 'BA(PPM)',
    'SR(PPM)', 'Y(PPM)', 'ZR(PPM)', 'NB(PPM)', 'LA(PPM)', 'CE(PPM)',
    'PR(PPM)', 'ND(PPM)', 'SM(PPM)', 'EU(PPM)', 'GD(PPM)', 'TB(PPM)',
    'DY(PPM)', 'HO(PPM)', 'ER(PPM)', 'YB(PPM)', 'LU(PPM)', 'HF(PPM)',
    'TA(PPM)', 'TH(PPM)'
]

# 元素缺失比例图只统计这 26 个 ppm 微量元素。
TRACE_ELEMENT_COLUMNS = [
    'RB(PPM)', 'V(PPM)', 'CR(PPM)', 'CO(PPM)', 'NI(PPM)', 'BA(PPM)',
    'SR(PPM)', 'Y(PPM)', 'ZR(PPM)', 'NB(PPM)', 'LA(PPM)', 'CE(PPM)',
    'PR(PPM)', 'ND(PPM)', 'SM(PPM)', 'EU(PPM)', 'GD(PPM)', 'TB(PPM)',
    'DY(PPM)', 'HO(PPM)', 'ER(PPM)', 'YB(PPM)', 'LU(PPM)', 'HF(PPM)',
    'TA(PPM)', 'TH(PPM)'
]

# 方法顺序 & 配色：当前 get_methods() 实际只启用下面 3 个方法。
METHOD_ORDER = ['KNN', 'MICE', 'MissForest']
PLOT_METHOD_ORDER = ['KNN', 'MICE', 'MissForest']
METHOD_COLORS = {
    'Mean':       '#8DA0CB',
    'Median':     '#66C2A5',
    # 'KNN (k=5)':  '#A6D854',
    'KNN':        '#D9D9D9',
    'MICE':       '#A6A6A6',
    'MissForest': '#5F7F95',
}
MISSFOREST_EDGE_COLOR = '#4F697A'


# ============================================================================
# 数据加载
# ============================================================================

def prepare_chemical_data(df, columns, row_missing_threshold=0.8):
    """中文注释：从现代总表子集提取 36 个元素列，并保留缺失不超过 80% 的样品。"""
    chemical_data = df.reindex(columns=columns).apply(pd.to_numeric, errors='coerce')
    min_non_null = int(len(columns) * (1 - row_missing_threshold))
    chemical_data = chemical_data.dropna(thresh=min_non_null).reset_index(drop=True)
    return chemical_data


def list_tectonic_frames():
    """中文注释：读取现代合并总表，并按 TECTONIC SETTING 动态拆分为评价子集。"""
    full_data = pd.read_csv(FULL_DATA_CSV, low_memory=False)
    if LABEL_COLUMN not in full_data.columns:
        raise KeyError(f"缺少标签列: {LABEL_COLUMN}")
    frames = []
    for setting_name, setting_df in full_data.groupby(LABEL_COLUMN, dropna=True):
        frames.append((str(setting_name), setting_df.reset_index(drop=True)))
    return sorted(frames, key=lambda item: item[0])

# ============================================================================
# MissForest 实现：基于 IterativeImputer 的迭代随机森林插补
# ============================================================================

def missforest_imputation(X_train, X_test):
    """
    迭代版 MissForest：
      - 使用训练折拟合 IterativeImputer，基学习器为 RandomForestRegressor
      - 用训练折学到的迭代插补关系 transform 测试折
      - 只改变测试折中缺失位置，后续评价仍只在人工 mask=True 的位置计算
    """
    X_train_fit, X_test_fit = _prepare_for_sklearn_imputer(X_train, X_test)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    imputer = IterativeImputer(
        estimator=rf,
        max_iter=15,
        initial_strategy='median',
        imputation_order='ascending',
        skip_complete=False,
        random_state=42,
    )
    return pd.DataFrame(
        imputer.fit(X_train_fit).transform(X_test_fit),
        columns=X_test.columns, index=X_test.index
    )


# ============================================================================
# 6 种插补方法（统一接口：method(X_train, X_test) -> X_imputed）
# ============================================================================

def _prepare_for_sklearn_imputer(X_train, X_test):
    """处理训练折全为空的列，避免 sklearn 插补器丢列或报错"""
    X_train_fit = X_train.copy()
    X_test_fit = X_test.copy()
    empty_cols = X_train_fit.columns[X_train_fit.notna().sum() == 0]
    if len(empty_cols) > 0:
        # 训练折完全没有观测值的元素无法学习分布，标准化空间下用 0 作为保底填充值
        X_train_fit.loc[:, empty_cols] = 0.0
    return X_train_fit, X_test_fit


def _simple_impute(X_train, X_test, strategy):
    X_train_fit, X_test_fit = _prepare_for_sklearn_imputer(X_train, X_test)
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(
        imputer.fit(X_train_fit).transform(X_test_fit),
        columns=X_test.columns, index=X_test.index
    )


def _knn_impute(X_train, X_test, n_neighbors):
    X_train_fit, X_test_fit = _prepare_for_sklearn_imputer(X_train, X_test)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    return pd.DataFrame(
        imputer.fit(X_train_fit).transform(X_test_fit),
        columns=X_test.columns, index=X_test.index
    )


def _mice_impute(X_train, X_test):
    X_train_fit, X_test_fit = _prepare_for_sklearn_imputer(X_train, X_test)
    imputer = IterativeImputer(
        estimator=BayesianRidge(), max_iter=10, random_state=42
    )
    return pd.DataFrame(
        imputer.fit(X_train_fit).transform(X_test_fit),
        columns=X_test.columns, index=X_test.index
    )


def get_methods():
    return {
        # 'Mean': lambda Xtr, Xte: _simple_impute(Xtr, Xte, 'mean'),
        # 'Median': lambda Xtr, Xte: _simple_impute(Xtr, Xte, 'median'),
        # 'KNN (k=5)': lambda Xtr, Xte: _knn_impute(Xtr, Xte, 5),
        'KNN': lambda Xtr, Xte: _knn_impute(Xtr, Xte, 20),
        'MICE': _mice_impute,
        'MissForest': missforest_imputation,
    }


# ============================================================================
# 评价（关键：仅在掩码位置上算）
# ============================================================================

def evaluate_at_mask(X_true, X_imputed, mask):
    """仅在 mask=True 的位置上计算 MAE / RMSE / R²"""
    true_vals, pred_vals = get_valid_masked_values(X_true, X_imputed, mask)
    if len(true_vals) == 0:
        return np.nan, np.nan, np.nan
    return calculate_metrics(true_vals, pred_vals)


def get_valid_masked_values(X_true, X_imputed, mask):
    """取出被人为掩码且 true/pred 都有效的评价点"""
    true_vals = X_true.values[mask]
    pred_vals = X_imputed.values[mask]
    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    return true_vals[valid], pred_vals[valid]


def calculate_metrics(true_vals, pred_vals):
    """基于一组评价点计算 MAE / RMSE / R²"""
    if len(true_vals) == 0:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    r2 = r2_score(true_vals, pred_vals)
    return mae, rmse, r2


def evaluate_per_element(X_true, X_imputed, mask):
    """按元素列分别计算 RMSE，用于元素级胜出统计"""
    out = {}
    for j, col in enumerate(X_true.columns):
        col_mask = mask[:, j]
        if col_mask.sum() == 0:
            out[col] = np.nan
            continue
        t = X_true.values[col_mask, j]
        p = X_imputed.values[col_mask, j]
        v = np.isfinite(t) & np.isfinite(p)
        if v.sum() == 0:
            out[col] = np.nan
        else:
            out[col] = np.sqrt(mean_squared_error(t[v], p[v]))
    return out


# ============================================================================
# 单个构造环境的 5 折 CV 主循环
# ============================================================================

def run_comparison_for_setting(setting_name, df, value_cache):
    methods = get_methods()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    if len(df) < N_FOLDS:
        print(f"  [跳过] {setting_name}: 样品数 {len(df)} 小于 {N_FOLDS} 折")
        return pd.DataFrame()

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        print(f"\n  [{setting_name}] Fold {fold}/{N_FOLDS} | 训练 {len(train_idx)} | 测试 {len(test_idx)}")

        X_train_raw = df.iloc[train_idx].copy()
        X_test_raw = df.iloc[test_idx].copy()

        # 每折内单独 fit 标准化器，避免用测试折信息造成数据泄露
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=df.columns, index=X_train_raw.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            columns=df.columns, index=X_test_raw.index
        )

        # 生成可复现的掩码（关键：只掩码原本有值的位置）
        rng = np.random.RandomState(RANDOM_SEED + fold)
        random_mask = rng.rand(*X_test.shape) < MASK_RATIO
        mask = random_mask & X_test.notna().values
        observed_count = X_test.notna().values.sum()
        actual_ratio = mask.sum() / observed_count if observed_count > 0 else np.nan
        print(f"           实际掩码率（在有值位置中）: {actual_ratio:.2%}")

        X_test_missing = X_test.mask(
            pd.DataFrame(mask, index=X_test.index, columns=X_test.columns)
        )

        for method_name, method_fn in methods.items():
            try:
                X_imputed = method_fn(X_train, X_test_missing)
                mae, rmse, r2 = evaluate_at_mask(X_test, X_imputed, mask)
                true_vals, pred_vals = get_valid_masked_values(X_test, X_imputed, mask)
                value_cache[method_name]['true'].append(true_vals)
                value_cache[method_name]['pred'].append(pred_vals)

                fold_results.append({
                    'TectonicSetting': setting_name,
                    'Fold': fold,
                    'Method': method_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'N_Eval': len(true_vals),
                    'ActualMaskRatio': actual_ratio,
                })
                print(f"    {method_name:>12s} | N={len(true_vals):6d}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

            except Exception as e:
                print(f"    [失败] {method_name}: {e}")

    return pd.DataFrame(fold_results)


def build_overall_results(value_cache):
    """基于所有构造环境、所有折、所有掩码点合并后的 true/pred 计算总体指标"""
    rows = []
    for method_name in METHOD_ORDER:
        true_parts = value_cache[method_name]['true']
        pred_parts = value_cache[method_name]['pred']
        if true_parts:
            true_vals = np.concatenate(true_parts)
            pred_vals = np.concatenate(pred_parts)
        else:
            true_vals = np.array([])
            pred_vals = np.array([])
        mae, rmse, r2 = calculate_metrics(true_vals, pred_vals)
        rows.append({
            'Method': method_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'N_Eval': len(true_vals),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 出版级绘图设置
# ============================================================================

def setup_plot_style():
    """Nature / Science 期刊风格"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'mathtext.default': 'regular',
    })


# ============================================================================
# 主图：3 个子图（仅保留 MAE / RMSE / R² 指标柱状图）
# ============================================================================

def _blend_color(color, target, amount):
    """把颜色轻微混合到目标色，用于控制柱内弱渐变。"""
    base = np.array(to_rgb(color))
    target_rgb = np.array(to_rgb(target))
    return tuple(base * (1.0 - amount) + target_rgb * amount)


def _add_subtle_bar_gradient(ax, bars, colors):
    """给柱子加入很弱的上下渐变，保持低饱和、非立体化效果。"""
    for bar, color in zip(bars, colors):
        height = bar.get_height()
        if not np.isfinite(height) or height <= 0:
            continue

        x0 = bar.get_x()
        x1 = x0 + bar.get_width()
        y0 = 0
        y1 = height
        top_color = _blend_color(color, '#FFFFFF', 0.12)
        bottom_color = _blend_color(color, '#000000', 0.05)
        gradient = np.linspace(bottom_color, top_color, 128).reshape(128, 1, 3)
        image = ax.imshow(
            gradient,
            extent=[x0, x1, y0, y1],
            origin='lower',
            aspect='auto',
            interpolation='bicubic',
            zorder=bar.get_zorder() + 0.1,
        )
        image.set_clip_path(bar)
        bar.set_facecolor((0, 0, 0, 0))


def plot_main_figure(results_df, save_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 4.05))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.095, right=0.985, top=0.78, bottom=0.30, wspace=0.22)

    # 三联柱状图：整体保持白底、低饱和“两灰一蓝”。
    plot_df = results_df.set_index('Method').loc[PLOT_METHOD_ORDER]
    metrics = [
        ('MAE', 'MAE', 'MAE ↓', False, 'a'),
        ('RMSE', 'RMSE', 'RMSE ↓', False, 'b'),
        ('R2', r'R$^2$', r'R$^2$ ↑', True, 'c'),
    ]

    legend_labels = ['KNN', 'MICE', 'missForest']
    colors = [METHOD_COLORS[m] for m in PLOT_METHOD_ORDER]

    for ax, (col, ylabel, title, higher_better, subletter) in zip(axes, metrics):
        means = plot_df[col].values
        x = np.array([0.0, 0.58, 1.16])
        edge_colors = ['#CFCFCF', '#8F8F8F', MISSFOREST_EDGE_COLOR]
        bars = ax.bar(
            x,
            means,
            width=0.36,
            color=colors,
            edgecolor=edge_colors,
            linewidth=0.55,
            zorder=3,
        )
        _add_subtle_bar_gradient(ax, bars, colors)

        # 数值标注统一三位小数；仅 missForest 最优值加粗。
        y_min = np.nanmin(means)
        y_max = np.nanmax(means)
        y_floor = 0 if y_min >= 0 else y_min
        y_range = max(y_max - y_floor, 1e-6)
        y_pad = max(y_range * 0.18, 0.035 if col == 'R2' else 0.016)
        y_text_offset = max(y_range * 0.030, 0.007 if col == 'R2' else 0.004)
        ax.set_ylim(y_floor, y_max + y_pad)
        ax.set_xlim(-0.34, 1.50)

        best_value = np.nanmax(means) if higher_better else np.nanmin(means)
        for j, value in enumerate(means):
            is_missforest_best = (
                PLOT_METHOD_ORDER[j] == 'MissForest'
                and np.isclose(value, best_value, rtol=0, atol=1e-12)
            )
            ax.text(
                x[j],
                value + y_text_offset,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=7.8,
                fontweight='bold' if is_missforest_best else 'normal',
                color='#111111' if is_missforest_best else '#333333',
                zorder=5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(legend_labels, rotation=0, ha='center')
        y_unit = ' (standardized)' if col != 'R2' else ' (standardized)'
        ax.set_ylabel(f'{ylabel}{y_unit}')
        ax.set_title(title, fontsize=9, pad=4)

        # 中文注释：主图恢复轻量坐标轴样式，仅保留左、下边框。
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000')
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='x', length=0, pad=5)
        ax.tick_params(axis='y', colors='#222222', width=0.7, length=3)
        if col == 'MAE':
            # MAE 面板固定刻度到 0.30，彻底去掉顶部 0.35 标注。
            ax.set_yticks(np.arange(0.0, 0.31, 0.05))
        ax.grid(axis='y', linestyle='-', color='#E7E7E7', alpha=1.0, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.set_facecolor('white')

        ax.text(
            -0.16,
            1.055,
            subletter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            ha='left',
            va='top',
        )
        
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [图] {save_path}")

def build_element_missing_ratio():
    """基于完整总表统计 26 个 ppm 元素的缺失比例。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_df = pd.read_csv(
        FULL_DATA_CSV,
        usecols=lambda col: col in TRACE_ELEMENT_COLUMNS
    )
    full_df = full_df.reindex(columns=TRACE_ELEMENT_COLUMNS)
    full_df = full_df.apply(pd.to_numeric, errors='coerce')

    total_count = len(full_df)
    rows = []
    for column in TRACE_ELEMENT_COLUMNS:
        missing_count = int(full_df[column].isna().sum())
        missing_ratio = missing_count / total_count * 100 if total_count > 0 else np.nan
        element_name = column.replace('(PPM)', '').title()
        rows.append({
            'Element': element_name,
            'Column': column,
            'MissingCount': missing_count,
            'TotalCount': total_count,
            'MissingRatio': missing_ratio,
        })

    missing_df = pd.DataFrame(rows)
    missing_df = missing_df.sort_values('MissingRatio', ascending=False).reset_index(drop=True)
    missing_df.to_csv(ELEMENT_MISSING_RATIO_CSV, index=False)
    return missing_df


def plot_element_missing_ratio(missing_df, save_path):
    """绘制 26 个 ppm 元素缺失比例横向 lollipop chart。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_plot_style()

    highlight_color = '#8A5145'
    regular_color = '#2F3A40'
    # 高亮颜色表示缺失率最高的前 5 个元素。
    colors = [highlight_color if i < 5 else regular_color for i in range(len(missing_df))]

    fig, ax = plt.subplots(figsize=(6.2, 6.7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    y = np.arange(len(missing_df))
    x = missing_df['MissingRatio'].to_numpy()
    x_max = np.nanmax(x) if len(x) else 0
    label_gap = 0.9
    right_padding = 6.0

    for i, (ratio, color) in enumerate(zip(x, colors)):
        ax.hlines(i, 0, ratio, color=color, linewidth=1.15, alpha=0.96)
        ax.scatter(ratio, i, s=28, color=color, edgecolor='white', linewidth=0.45, zorder=3)
        ax.text(ratio + label_gap, i, f'{ratio:.1f}%', va='center', ha='left',
                fontsize=7.4, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(missing_df['Element'])
    ax.invert_yaxis()
    ax.set_xlabel('Missing-data proportion (%)')
    ax.set_ylabel('')
    # 不设置标题，保持附图版式更简洁。

    ax.set_xlim(0, min(105, x_max + label_gap + right_padding))
    ax.grid(axis='x', linestyle='--', color='#EDEDED', alpha=0.55, linewidth=0.45)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4A4A4A')
    ax.spines['bottom'].set_color('#4A4A4A')
    ax.tick_params(axis='both', colors='#333333', width=0.7, length=3)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  [图] {save_path}")

# ============================================================================
# 主流程
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  缺失值插补方法对比 —— 按构造环境分别 5 折 CV × 10% 人为掩码")
    print(f"  方法: {' | '.join(METHOD_ORDER)}")
    print("=" * 72)

    print(f"\n[数据总表] {FULL_DATA_CSV}")
    tectonic_frames = list_tectonic_frames()
    if not tectonic_frames:
        raise ValueError(f"现代总表中没有可用的 {LABEL_COLUMN} 分组")
    print(f"  找到 {len(tectonic_frames)} 个构造环境分组")

    all_fold_results = []
    value_cache = {
        method_name: {'true': [], 'pred': []}
        for method_name in METHOD_ORDER
    }

    # 中文注释：按构造环境分组分别做交叉验证与插补评价，不跨类别合并训练。
    for setting_name, setting_frame in tectonic_frames:
        print("\n" + "-" * 72)
        print(f"[构造环境] {setting_name}")

        df = prepare_chemical_data(setting_frame, CHEMICAL_COLUMNS)
        print(f"  样品 × 元素: {df.shape}")
        print(f"  整体缺失率:  {df.isnull().values.mean():.2%}")
        print(f"  各元素缺失率前 5:")
        for col, pct in df.isnull().mean().nlargest(5).items():
            print(f"    {col:14s} {pct:.1%}")

        fold_df = run_comparison_for_setting(setting_name, df, value_cache)
        if not fold_df.empty:
            all_fold_results.append(fold_df)

    if not all_fold_results:
        raise RuntimeError("没有生成任何有效评价结果，请检查各构造环境样品数量和元素列。")

    results_by_fold = pd.concat(all_fold_results, ignore_index=True)
    results_by_fold.to_csv(RESULTS_BY_FOLD_CSV, index=False)

    # 每类构造环境下各方法的折间平均结果
    results_by_setting = results_by_fold.groupby(['TectonicSetting', 'Method']).agg(
        MAE_mean=('MAE', 'mean'),
        MAE_std=('MAE', 'std'),
        RMSE_mean=('RMSE', 'mean'),
        RMSE_std=('RMSE', 'std'),
        R2_mean=('R2', 'mean'),
        R2_std=('R2', 'std'),
        N_Eval=('N_Eval', 'sum'),
        ActualMaskRatio_mean=('ActualMaskRatio', 'mean'),
    ).reset_index()
    results_by_setting['Method'] = pd.Categorical(
        results_by_setting['Method'], categories=METHOD_ORDER, ordered=True
    )
    results_by_setting = results_by_setting.sort_values(['TectonicSetting', 'Method'])
    results_by_setting.to_csv(RESULTS_BY_SETTING_CSV, index=False)

    # 总体指标：按所有实际被掩码评价点合并 true/pred 后重新计算
    results_overall = build_overall_results(value_cache)
    results_overall.to_csv(RESULTS_OVERALL_CSV, index=False)

    print("\n" + "=" * 72)
    print("  总体汇总（所有构造环境、所有折、所有掩码点合并计算）")
    print("=" * 72)
    for _, row in results_overall.iterrows():
        print(f"  {row['Method']:>12s} | N = {int(row['N_Eval']):7d}  "
              f"MAE = {row['MAE']:.4f}  "
              f"RMSE = {row['RMSE']:.4f}  "
              f"R2 = {row['R2']:.4f}")

    # 相对第二名的提升幅度（用于正文写"领先 X%"）
    mf_rmse = results_overall[results_overall['Method'] == 'MissForest']['RMSE'].iloc[0]
    others = results_overall[results_overall['Method'] != 'MissForest'].sort_values('RMSE')
    second_best = others.iloc[0]
    improvement = (second_best['RMSE'] - mf_rmse) / second_best['RMSE'] * 100
    print(f"\n  -> MissForest RMSE 较第二名（{second_best['Method']}）低 {improvement:.1f}%")

    # 绘图：主图直接使用总体结果
    print("\n[绘图]")
    plot_main_figure(
        results_overall,
        MAIN_FIGURE_PNG
    )
    missing_df = build_element_missing_ratio()
    plot_element_missing_ratio(
        missing_df,
        ELEMENT_MISSING_RATIO_PNG
    )
    print(f"  [插补评价图] {MAIN_FIGURE_PNG}")
    print(f"  [元素缺失比例图] {ELEMENT_MISSING_RATIO_PNG}")

    print(f"\n[完成] 全部输出 -> {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
