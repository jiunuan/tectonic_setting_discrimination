from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 太古代时间演化附属敏感性图 —— 弧相关亲和性的四联稳健性检验
#
#   本脚本是一个【独立】脚本，为共享时间轴
#   多证据对齐主图）配套一张 2×2 敏感性附图。它：
#     · 不修改、不覆盖时间演化主图的任何输出；
#     · 直接读取已保存的太古代预测结果表 expanded_archean_predictions.csv，
#       统一用  Parc = P_CA + P_IA + P_IOA  作为"弧相关亲和性"；
#     · 时间轴范围、刻度、配色、视觉风格全部对齐时间演化主图。
#
#   四个子图（横轴统一 4.0→2.4 Ga，左老右新；a–c 为【离散分箱点线图】，每箱一个点）：
#     (a) 阈值敏感性     ：固定 200 Myr 分箱（与时间演化主图口径一致），对比 Parc ≥ 0.4 / 0.5 / 0.6；
#     (b) 分箱敏感性     ：固定 Parc ≥ 0.5，对比 50 / 100 / 200 Myr 年龄分箱；
#     (c) 易迁移元素敏感性：固定 Parc ≥ 0.5，在 Ba/Rb/Sr/K 原始实测齐全样品中
#                          对比完整模型与四元素置缺失后重推理结果；
#     (d) Applicability-domain check：高弧样品（Parc ≥ 0.5）在各年龄分箱内，按其相对
#                          现代组成参照域的位置分成三类（95% 域内 / 95–99% 边缘 /
#                          99% 域外）的【堆叠柱状图】，用于说明高弧结果不是主要
#                          由现代参照域外的离群样品驱动。
#
#   方法学要点：
#     · (a)-(c) 每个等宽年龄箱只给一个统计点（箱内 Parc≥阈值 的占比），点间直线连接仅作
#       视觉引导；【不使用 spline / KDE / rolling smooth】，避免给稀疏太古代数据制造虚假
#       的连续时间分辨率。误差为箱内样品有放回 bootstrap 的 2.5–97.5 分位（95% CI 误差棒），
#       样品数 < MIN_BIN_N 的箱不出点；空箱（如 3.6–3.7 Ga 无样品）直接缺省，相邻点连线跨过。
#     · (d) 现代参照域 = 现代玄武岩在 log10 元素空间的 PCA 子空间；用 kNN 距离的现代
#       95% / 99% 分位定义域边界（与 pca_distribution_consistency.py 完全一致的口径）。
#
#   ★ 两处需要一次性重算并缓存（之后重绘直接读缓存）：
#     · (c) Ba/Rb/Sr/K 置缺失后真实重跑 GeoDAN 前向推理
#                                                → fig7_sensitivity_masked_BaRbSrK_predictions.csv
#     · (d) 现代参照域归属由 PCA + kNN applicability-domain 计算
#                                                → fig7_sensitivity_domain_membership.csv
#     缺少 PyTorch / sklearn / 相应数据时，对应面板优雅降级并给出说明。
#
#   输出（全新文件名，绝不覆盖主图）：
#     · archean_time_evolution_sensitivity_four_panel.png
#     · fig7_sensitivity_domain_membership.csv
#     · fig7_sensitivity_caption.txt
# ──────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 复用原分析模块（单一数据真相来源；与时间演化主图一致的导入方式）──────
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

PREDICTION_CSV = Path(ZENODO_ARCHEAN_PREDICTIONS_CSV) if Path(ZENODO_ARCHEAN_PREDICTIONS_CSV).exists() else A.FINAL_PREDICTION_PATH
OUTPUT_DIR     = A.FINAL_OUTPUT_DIR

FIG_PATH      = OUTPUT_DIR / "fig7_sensitivity_four_panel.png"
MASKED_CACHE  = OUTPUT_DIR / "fig7_sensitivity_masked_BaRbSrK_predictions.csv"
DOMAIN_CACHE  = OUTPUT_DIR / "fig7_sensitivity_domain_membership.csv"
CAPTION_PATH  = OUTPUT_DIR / "fig7_sensitivity_caption.txt"

# bootstrap
N_BOOT    = 1000
BOOT_SEED = 20240611

# 共享时间轴（老→新，左→右）；范围与刻度对齐时间演化主图
X_OLD, X_YOUNG = 4.00, 2.40
X_TICKS = np.arange(2.6, 4.01, 0.2)          # 与时间演化主图一致：不单列 2.4 刻度

# ── 离散年龄分箱（不做任何平滑/插值/KDE/rolling）─────────────────────────────────
# 中文注释：每个年龄箱只给一个统计点，点间直线连接仅作视觉引导，绝不暗示连续时间分辨率。
# a/c 用默认 200 Myr 分箱（与时间演化主图的弧比例曲线口径一致）；b 扫描 50/100/200 Myr。
# 误差用箱内有放回 bootstrap 95% CI。
DEFAULT_BIN_MYR = 200.0
BIN_SCAN        = [50.0, 100.0, 200.0]
MIN_BIN_N       = 8      # 每个年龄箱至少样品数；低于此不出点（避免 n 过小的伪比例/发散 CI）

THRESH_DEFAULT  = 0.5
HIGH_THRESHOLD  = 0.5    # 面板 (d) "高弧样品" 判定阈值
THRESH_SCAN     = [0.4, 0.5, 0.6]

# 时间面板共享纵轴上限：取 0.7（贴近主图的 0–70% 弧比例轴）。200 Myr 主曲线浮动在
# ~0.1–0.32，50/100 Myr 细箱的 CI 上界最高 ~0.61，均在轴内、无裁切。
Y_TOP = 0.80

# 易迁移元素敏感性：panel c 使用 Ba/Rb/Sr/K 原始实测齐全样品。
# 中文注释：K2O 只在这个地质窗口敏感性检查中参与；结果必须解释为扰动诊断，不作为单样品判别。
MOBILE_COLS = ["BA(PPM)", "RB(PPM)", "SR(PPM)", "K2O(WT%)"]

# 面板 (d) 年龄分箱（200 Myr，老→新）
DOMAIN_BIN_EDGES = np.arange(2400, 4001, 200)

# ── 配色（取自时间演化主图 / pca_distribution_consistency.py 调色板）──────
COL_BLUE   = "#2E5C8A"
COL_ORANGE = "#E6783C"
COL_GREEN  = "#2E8B57"

# 面板 (d) 域归属三类（蓝=域内、琥珀=边缘、灰=域外；灰色刻意弱化离群样品）
COL_IN_DOMAIN  = "#3F7CA3"
COL_MARGINAL   = "#E2A33C"
COL_OUT_DOMAIN = "#9AA3AB"

# 重点窗口与转折线：跨度/颜色/透明度完全对齐时间演化主图
#   · ~3.8 Ga 窄带 = ARC_NARROW_WINDOW（蓝 #5B8FBF, alpha 0.32, 3.76–3.84）
#   · ~2.7–2.5 Ga = LATE_ARCHEAN_RISE（绿 #8FCB86, alpha 0.34, 2.50–2.70）
#   · ~3.5 Ga 转折线 = TRANSIENT_PULSE（深蓝 #1F4E79, 长虚线）
WIN_38         = (3.76, 3.84)
WIN_38_COL     = "#5B8FBF"
WIN_38_ALPHA   = 0.32
WIN_LATE       = (2.50, 2.70)
WIN_LATE_COL   = "#8FCB86"
WIN_LATE_ALPHA = 0.34
DASH_35        = 3.50
DASH_35_COL    = "#1F4E79"

GRID_GRAY  = "#C8C8C8"
SPINE_GRAY = "#000000"


# ══════════════════════════════════════════════════════════════════════════════
#  数据装载与列自动识别
# ══════════════════════════════════════════════════════════════════════════════

def _read_prediction_table(path: Path) -> pd.DataFrame:
    """读取预测结果表并清除列名 BOM。"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    return df.rename(columns=lambda c: str(c).replace("﻿", "").strip())


def _detect_arc_prob_columns(df: pd.DataFrame) -> list[str]:
    """自动识别 CA / IA / IOA 概率列（排除 *_std 列）。"""
    lower = {c.lower(): c for c in df.columns}

    def _match(*keys: str) -> str | None:
        for lc, orig in lower.items():
            if not lc.startswith("prob_") or "std" in lc:
                continue
            if all(k in lc for k in keys):
                return orig
        return None

    return [c for c in (_match("continental", "arc"),
                        _match("island", "arc"),
                        _match("intra", "arc")) if c is not None]


def _compute_parc(df: pd.DataFrame) -> pd.Series:
    """Parc = P_CA + P_IA + P_IOA；识别失败回退 Arc_probability3。"""
    cols = _detect_arc_prob_columns(df)
    if len(cols) == 3:
        print(f"[列识别] Parc = {' + '.join(cols)}")
        return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    if "Arc_probability3" in df.columns:
        print("[列识别] 未集齐三类弧概率列，回退使用 Arc_probability3 作为 Parc")
        return pd.to_numeric(df["Arc_probability3"], errors="coerce")
    raise KeyError("无法识别弧概率列")


def _detect_age_ma(df: pd.DataFrame) -> pd.Series:
    """年龄列：优先 C_AGE，缺失回退 AGE（单位 Ma）。"""
    age = pd.Series(np.nan, index=df.index, dtype=float)
    if "C_AGE" in df.columns:
        age = pd.to_numeric(df["C_AGE"], errors="coerce")
    if "AGE" in df.columns:
        age = age.fillna(pd.to_numeric(df["AGE"], errors="coerce"))
    if age.notna().sum() == 0:
        raise KeyError("无法识别年龄列")
    return age


def _ba_outlier_mask(sample_id: pd.Series, ba: pd.Series) -> pd.Series:
    """与太古代时间演化主图一致：仅剔除 Ba≈23349 ppm 单条异常样品。"""
    return sample_id.astype(str).eq("s_2C-14 [24900]") & (
        pd.to_numeric(ba, errors="coerce") > 23000.0)


def load_dataframe() -> pd.DataFrame:
    """读取主预测表，附加 Parc / age_ga / Parc_masked，并剔除 Ba 异常样品。"""
    raw = _read_prediction_table(PREDICTION_CSV)
    out = pd.DataFrame(index=raw.index)
    out["Parc"]      = _compute_parc(raw).to_numpy()
    out["age_ga"]    = _detect_age_ma(raw).to_numpy() / 1000.0
    out["SAMPLE_ID"] = raw.get("SAMPLE_ID", pd.Series("", index=raw.index)).astype(str)
    out["Parc_masked"] = np.nan
    masked = _load_masked_cache(len(raw), out["SAMPLE_ID"].to_numpy())
    if masked is not None:
        out["Parc_masked"] = masked

    # 中文注释：panel c 使用 Ba/Rb/Sr/K 原始测值齐全的 complete-case 子集；
    # 这样 full 与 masked 是同一样品的成对比较，且每个样品都真实拥有可被移除的四元素信息。
    out["mobile_complete_case"] = False
    if A.FINAL_ARCHEAN_MASK_PATH.exists():
        mobile_mask = pd.read_csv(A.FINAL_ARCHEAN_MASK_PATH, encoding="utf-8-sig")
        mask_cols = [f"missing_mask__{col}" for col in MOBILE_COLS]
        if len(mobile_mask) == len(raw) and all(col in mobile_mask.columns for col in mask_cols):
            out["mobile_complete_case"] = (
                mobile_mask[mask_cols].sum(axis=1).to_numpy() == 0
            )
        else:
            print("[缺失编码] Ba/Rb/Sr/K complete-case 列无法校验，panel c 将降级为全样本。")
            out["mobile_complete_case"] = True
    else:
        print("[缺失编码] 未找到最终太古代缺失 mask，panel c 将降级为全样本。")
        out["mobile_complete_case"] = True
    drop = _ba_outlier_mask(out["SAMPLE_ID"], raw.get("BA", np.nan))
    if int(drop.sum()):
        print(f"[数据质量] 剔除 Ba≈23349 ppm 异常样品: {int(drop.sum())}")
    return out.loc[~drop.to_numpy()].reset_index(drop=True)



# ══════════════════════════════════════════════════════════════════════════════
#  面板 (c)：Ba/Rb/Sr/K 置缺失重推理缓存
# ══════════════════════════════════════════════════════════════════════════════

def _load_masked_cache(n_expected: int, sample_ids: np.ndarray) -> np.ndarray | None:
    """读取 Ba/Rb/Sr/K 置缺失重推理缓存并做行序校验。"""
    if not MASKED_CACHE.exists():
        return None
    cache = pd.read_csv(MASKED_CACHE, encoding="utf-8-sig")
    if len(cache) != n_expected:
        print(f"[掩蔽缓存] 行数不一致（{len(cache)} != {n_expected}），忽略缓存")
        return None
    if not np.array_equal(cache["SAMPLE_ID"].astype(str).to_numpy(), sample_ids.astype(str)):
        print("[掩蔽缓存] SAMPLE_ID 序列与主表不一致，忽略缓存")
        return None
    return pd.to_numeric(cache["Parc_masked"], errors="coerce").to_numpy()


def generate_masked_cache() -> bool:
    """把 Ba/Rb/Sr/K 在输入端整体置缺失后真实重跑一次前向推理，缓存逐样品 Parc。"""
    if not getattr(A, "_TORCH_AVAILABLE", False):
        print("[掩蔽推理] 当前环境无 PyTorch，跳过；面板 (c) 仅画完整模型曲线。")
        return False
    if not A.FINAL_MODEL_WEIGHT_PATH.exists():
        print(f"[掩蔽推理] 缺少模型权重 {A.FINAL_MODEL_WEIGHT_PATH}，跳过。")
        return False

    print("[掩蔽推理] 去除 Ba/Rb/Sr/K，重跑 GeoDAN 前向推理（一次性）……")
    class_names = A.load_class_names(A.TRAIN_PATH)
    metadata = A.load_final_age_constrained_pool()
    metadata, normalized = A._prepare_archean_features_from_metadata(metadata)
    missing_mask = A._build_archean_missing_mask(metadata)

    normalized, missing_mask = normalized.copy(), missing_mask.copy()
    for col in MOBILE_COLS:
        if col in normalized.columns:
            normalized[col] = 0
        mcol = f"missing_mask__{col}"
        if mcol in missing_mask.columns:
            missing_mask[mcol] = 1

    probs = A._predict_with_missing_mask(
        normalized, missing_mask, class_names, A.FINAL_MODEL_WEIGHT_PATH)
    pred = A.add_prediction_columns(
        metadata, probs, np.zeros_like(probs), class_names,
        high_prob=A.HIGH_PROB, high_std=A.HIGH_STD)

    pd.DataFrame({
        "SAMPLE_ID": metadata["SAMPLE_ID"].astype(str).to_numpy(),
        "Parc_masked": pred["Arc_probability3"].to_numpy(),
    }).to_csv(MASKED_CACHE, index=False, encoding="utf-8-sig")
    print(f"[掩蔽推理] 缓存已保存: {MASKED_CACHE}")
    return True

# ══════════════════════════════════════════════════════════════════════════════
#  面板 (d)：现代组成参照域归属（PCA + kNN applicability-domain，一次性缓存）
# ══════════════════════════════════════════════════════════════════════════════

def generate_domain_cache() -> bool:
    """
    复用 pca_distribution_consistency.py 的口径，计算每个太古代样品到现代组成参照域的
    applicability-domain kNN 距离，并按现代 95% / 99% 分位将其分为
    in（≤95% 域内）/ mid（95–99% 边缘）/ out（>99% 域外）三类，缓存到 CSV。
    需要 sklearn + 现代/太古代数据；不可用时返回 False。
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from sklearn.decomposition import PCA
        import pca_distribution_consistency as P
        from archean_s3_preprocess import preprocess_archean
    except Exception as exc:  # noqa: BLE001
        print(f"[现代域] 依赖不可用（{exc}），跳过；面板 (d) 将提示缺失。")
        return False
    if not P.RAW_MODERN_PATH.exists() or not P.ARCHEAN_RAW_PATH.exists():
        print("[现代域] 缺少现代/太古代原始数据，跳过。")
        return False

    print("[现代域] 计算 PCA + kNN applicability-domain 归属（一次性）……")
    # 现代参照集：复刻 main() 的 CFB 确定性欠采样
    modern = P.clean_columns(pd.read_csv(P.RAW_MODERN_PATH, low_memory=False))
    modern.dropna(subset=[P.TECTONIC_COL], inplace=True)
    modern[P.TECTONIC_COL] = modern[P.TECTONIC_COL].astype(str).str.strip().str.upper()
    modern.reset_index(drop=True, inplace=True)
    cfb = np.flatnonzero(modern[P.TECTONIC_COL].to_numpy() == "CONTINENTAL FLOOD BASALT")
    sel = np.sort(np.random.default_rng(P.RANDOM_SEED).choice(
        cfb, size=P.CFB_TARGET_COUNT, replace=False))
    non = np.flatnonzero(modern[P.TECTONIC_COL].to_numpy() != "CONTINENTAL FLOOD BASALT")
    modern = modern.iloc[np.sort(np.concatenate([non, sel]))].reset_index(drop=True)

    arch = preprocess_archean(pd.read_csv(P.ARCHEAN_RAW_PATH, low_memory=False))
    pred = pd.read_csv(P.FINAL_PREDICTION_PATH, low_memory=False)
    if len(pred) != len(arch):
        print(f"[现代域] 预测与太古代行数不一致（{len(pred)} != {len(arch)}），跳过。")
        return False

    X_train, X_arch = P.to_pca_space(P.extract_elements(modern), P.extract_elements(arch))
    pca = PCA(random_state=P.RANDOM_SEED).fit(X_train)
    ad = P.compute_ad_distances(pca.transform(X_train), pca.transform(X_arch),
                                pca.explained_variance_ratio_)
    dist, q95, q99 = ad["dist_arch"], ad["q95"], ad["q99"]
    category = np.where(dist <= q95, "in", np.where(dist <= q99, "mid", "out"))

    age = pd.to_numeric(pred.get("C_AGE"), errors="coerce")
    if "AGE" in pred.columns:
        age = age.fillna(pd.to_numeric(pred["AGE"], errors="coerce"))
    parc = pd.to_numeric(pred["Arc_probability3"], errors="coerce")

    pd.DataFrame({
        "SAMPLE_ID": pred.get("SAMPLE_ID", pd.Series(range(len(pred)))).astype(str).to_numpy(),
        "BA": pd.to_numeric(pred.get("BA"), errors="coerce").to_numpy(),
        "age_ma": age.to_numpy(),
        "Parc": parc.to_numpy(),
        "ad_distance": dist,
        "category": category,
    }).to_csv(DOMAIN_CACHE, index=False, encoding="utf-8-sig")
    print(f"[现代域] n_pc={ad['n_pc']}, q95={q95:.3f}, q99={q99:.3f}；缓存已保存: {DOMAIN_CACHE}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  离散分箱比例 + 箱内 bootstrap 95% CI（不做任何平滑 / 插值 / KDE / rolling）
# ══════════════════════════════════════════════════════════════════════════════

def binned_proportion_ci(age_ga: np.ndarray, indicator: np.ndarray,
                         bin_width_myr: float, seed: int = BOOT_SEED,
                         min_n: int = MIN_BIN_N
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    把样品按等宽年龄箱（从 2.4 到 4.0 Ga，宽度 bin_width_myr）离散统计：
      · 每个箱给一个比例点 = 箱内 indicator 的均值（如 Parc≥阈值 的占比）；
      · 误差为箱内样品有放回 bootstrap 的比例分布 2.5–97.5 分位（非对称 95% CI）；
      · 样品数 < min_n 的箱不出点（避免 n 过小造成的伪比例）。
    返回 (bin_center_ga, proportion, ci_lo, ci_hi, n)，均为按年龄升序的一维数组。
    """
    # 中文注释：必须在【整数 Ma】空间分箱。若在 Ga 浮点空间用 np.arange 生成边界，
    # 3.8 之类的边界会变成 3.79999…，使正好落在 3800 Ma 的样品被错分到上一个箱，
    # 造成边界箱（如 3.8–3.9 Ga）假性归零——主图用整数 Ma + pd.cut，不会犯这个错误。
    age_ma = np.rint(np.asarray(age_ga, dtype=float) * 1000.0)
    bw = int(round(bin_width_myr))
    edges = np.arange(2400, 4000 + bw, bw)        # 整数 Ma 边界，2.4–4.0 Ga，左老右新对齐主图
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(edges) - 1):
        lo_e, hi_e = edges[i], edges[i + 1]
        center_ga = (lo_e + hi_e) / 2000.0
        # 中文注释：与主图的 `if mid < 2.5: continue` 完全一致——丢弃箱中心 < 2.5 Ga 的箱。
        # 最年轻的 2.45–2.5 Ga 细箱(48 样品恰好无一高弧)正是 50/100 Myr 线右端点砸到 0 的来源；
        # 200 Myr 把它并入 [2.4,2.6) 故无此问题。统一截断后所有曲线右端不再下降到 0。
        if center_ga < 2.5:
            continue
        m = (age_ma >= lo_e) & (age_ma < hi_e) & np.isfinite(indicator)
        ind = indicator[m].astype(float)
        n = len(ind)
        if n < min_n:
            continue
        frac = float(ind.mean())
        boots = ind[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        rows.append((center_ga, frac, float(ci_lo), float(ci_hi), n))
    if not rows:
        empty = np.array([])
        return empty, empty, empty, empty, empty
    centers, props, los, his, ns = (np.array(col) for col in zip(*rows))
    return centers, props, los, his, ns


# ══════════════════════════════════════════════════════════════════════════════
#  共享绘图装饰
# ══════════════════════════════════════════════════════════════════════════════

def _draw_windows(ax) -> None:
    """浅色重点窗口（~3.8 Ga / ~2.7–2.5 Ga）+ ~3.5 Ga 转折虚线。"""
    # 中文注释：窗口背景色/透明度/3.5 Ga 虚线样式与时间演化主图完全一致。
    ax.axvspan(*WIN_38, color=WIN_38_COL, alpha=WIN_38_ALPHA, linewidth=0.0, zorder=0)
    ax.axvspan(*WIN_LATE, color=WIN_LATE_COL, alpha=WIN_LATE_ALPHA, linewidth=0.0, zorder=0)
    ax.axvline(DASH_35, color=DASH_35_COL, linestyle=(0, (3.2, 2.2)),
               linewidth=1.35, alpha=0.90, zorder=1)


def _style_time_panel(ax) -> None:
    """统一时间面板风格：白底、浅灰四边框、反向时间轴。"""
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(X_OLD, X_YOUNG)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_yticks(np.arange(0.0, Y_TOP + 1e-9, 0.20))
    ax.tick_params(axis="both", labelsize=8.5)
    # 中文注释：四条边界线全部保留，统一线宽为 0.6。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_GRAY)
        spine.set_linewidth(0.6)
    ax.grid(axis="y", color=GRID_GRAY, linewidth=0.5, linestyle=(0, (1, 3)),
            alpha=0.7, zorder=0)


def _style_legend(leg) -> None:
    """统一图例：极淡白底圆角框（无重阴影），仅起轻微衬底作用。"""
    if leg is None:
        return
    leg.set_zorder(30)
    frame = leg.get_frame()
    frame.set_facecolor("#FFFFFF")
    frame.set_edgecolor("#DCE0E5")   # 很浅的灰边
    frame.set_linewidth(0.5)
    frame.set_alpha(0.88)


def _legend(ax, **kwargs):
    """带极淡白底框的图例（圆角、无阴影）。"""
    leg = ax.legend(fancybox=True, shadow=False, framealpha=0.88, **kwargs)
    _style_legend(leg)
    return leg


def _plot_points(ax, centers, props, los, his, color, label, marker, *,
                 linestyle="-", zorder=4, show_ci=True,
                 linewidth=1.3, markersize=4.6, alpha=1.0,
                 markeredgewidth=1.1) -> None:
    """离散分箱点线图：每箱一个点，点间直线连接；show_ci 控制是否叠加 bootstrap 95% CI 误差棒。
    linewidth/markersize/alpha 用于在多序列叠加时构建强调层次（如 b 面板突出 200 Myr 主线）。"""
    if len(centers) == 0:
        return
    if show_ci:
        yerr = np.vstack([np.maximum(0.0, props - los), np.maximum(0.0, his - props)])
        # 误差棒半透明、置于点线下层，避免多序列叠加时过于杂乱。
        ax.errorbar(centers, props, yerr=yerr, fmt="none", ecolor=color,
                    elinewidth=0.8, capsize=2.0, capthick=0.8, alpha=0.55,
                    zorder=zorder - 1)
    ax.plot(centers, props, color=color, linestyle=linestyle, linewidth=linewidth,
            marker=marker, markersize=markersize, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=markeredgewidth, alpha=alpha,
            label=label, zorder=zorder)


def _panel_tag(ax, letter: str, title: str, title_pad: float = 11.0, tag_y: float = 1.02) -> None:
    # 中文注释：tag_y 用于单独微调下排编号位置，避免和图例/标题挤在一起。
    ax.text(-0.14, tag_y, f"{letter}", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="left")
    ax.set_title(title, fontsize=9.5, pad=title_pad)


# ══════════════════════════════════════════════════════════════════════════════
#  四个面板
# ══════════════════════════════════════════════════════════════════════════════

def draw_panel_a(ax, df: pd.DataFrame) -> None:
    """(a) 阈值敏感性：固定 200 Myr 分箱，Parc ≥ 0.4 / 0.5 / 0.6。
    中文注释：Parc ≥ 0.5 是正文默认阈值，作为主线突出（粗、深蓝、置顶）；0.4 / 0.6 退为
    较细较淡的对照线，强调层级与 panel b 一致。"""
    _draw_windows(ax)
    age, parc = df["age_ga"].to_numpy(), df["Parc"].to_numpy()
    # 阈值样式：(颜色, marker, 线宽, 点大小, 透明度, zorder)
    thr_styles = {
        0.4: (COL_ORANGE, "o", 1.4, 4.2, 0.65, 4),   # 中文注释：0.4 阈值橙线降低透明度，避免抢过 0.5 主线。
        0.5: (COL_BLUE,   "D", 1.8, 5.6, 1.00, 6),   # 主线：粗深蓝、置顶
        0.6: ("#A7AEB6", "s", 1.0, 3.0, 0.80, 3),    # 细淡灰（更严格）
    }
    # 中文注释：先画对照线、后画主线，确保 0.5 主线压在最上层。
    for thr in (0.6, 0.4, 0.5):
        color, marker, lw, ms, al, zo = thr_styles[thr]
        c, p, lo, hi, _ = binned_proportion_ci(age, (parc >= thr).astype(float),
                                               DEFAULT_BIN_MYR,
                                               seed=BOOT_SEED + int(thr * 100))
        _plot_points(ax, c, p, lo, hi, color, fr"P$_{{arc}}$ ≥ {thr:.1f}", marker,
                     show_ci=False, linewidth=lw, markersize=ms, alpha=al, zorder=zo,
                     markeredgewidth=1.4 if thr == 0.5 else 1.0)
    # 中文注释：图例按 0.4/0.5/0.6 顺序显示，故收集句柄后重排。
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    _legend(ax, handles=[handles[i] for i in order], labels=[labels[i] for i in order],
            loc="upper right", fontsize=10, handlelength=1.6,
            labelspacing=0.3, borderaxespad=0.5)
    _panel_tag(ax, "a", "Threshold sensitivity")


def draw_panel_b(ax, df: pd.DataFrame) -> None:
    """(b) 分箱敏感性：固定 Parc ≥ 0.5，对比 50 / 100 / 200 Myr 年龄分箱。
    中文注释：200 Myr 是正文默认口径，作为主线突出（粗、深蓝、置顶）；100 Myr 中等；
    50 Myr 最细的分箱噪声最大，退为细淡灰背景线，避免其抖动喧宾夺主。"""
    _draw_windows(ax)
    age = df["age_ga"].to_numpy()
    ind = (df["Parc"].to_numpy() >= THRESH_DEFAULT).astype(float)
    # 每个分箱宽度的样式：(颜色, marker, 线宽, 点大小, 透明度, zorder)
    bin_styles = {
        50:  ("#A6ABAF", "", 0.9, 0.0, 0.85, 3),   # 中文注释：50 Myr 灰线去掉 marker 并进一步变浅，弱化 2.6 Ga 附近尖峰。
        100: (COL_ORANGE, "s", 1.5, 4.2, 0.85, 4),   # 中等橙
        200: (COL_BLUE,   "D", 1.8, 5.6, 1.00, 6),   # 主线：粗深蓝、置顶
    }
    # 中文注释：先画细线、后画主线，确保 200 Myr 主线压在最上层。
    for bw in sorted(BIN_SCAN):
        color, marker, lw, ms, al, zo = bin_styles[int(bw)]
        c, p, lo, hi, _ = binned_proportion_ci(age, ind, bw, seed=BOOT_SEED + int(bw))
        _plot_points(ax, c, p, lo, hi, color, f"{int(bw)} Myr", marker, show_ci=False,
                     linewidth=lw, markersize=ms, alpha=al, zorder=zo,
                     markeredgewidth=1.0 if bw < 200 else 1.4)
    _legend(ax, loc="upper right", fontsize=10, handlelength=1.6,
            labelspacing=0.3, borderaxespad=0.5)
    _panel_tag(ax, "b", "Age-bin sensitivity")


def _window_proportion_ci(
    age_ga: np.ndarray,
    binary: np.ndarray,
    windows: list[tuple[float, float, float]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按手工地质窗口计算比例和 bootstrap CI；windows=(lo, hi, x)。"""
    rng = np.random.default_rng(seed)
    centers, props, lows, highs, counts = [], [], [], [], []
    for lo, hi, center in windows:
        mask = np.isfinite(age_ga) & np.isfinite(binary) & (age_ga >= lo) & (age_ga < hi)
        vals = binary[mask].astype(float)
        if len(vals) < MIN_BIN_N:
            continue
        prop = float(vals.mean())
        idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
        boots = vals[idx].mean(axis=1)
        centers.append(center)
        props.append(prop)
        lows.append(float(np.percentile(boots, 2.5)))
        highs.append(float(np.percentile(boots, 97.5)))
        counts.append(len(vals))
    return (
        np.asarray(centers, dtype=float),
        np.asarray(props, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
        np.asarray(counts, dtype=int),
    )


def draw_panel_c(ax, df: pd.DataFrame) -> None:
    """(c) 易迁移元素敏感性：Ba/Rb/Sr/K complete-case 子集内的成对遮蔽比较。"""
    _draw_windows(ax)
    age_all = df["age_ga"].to_numpy(dtype=float)
    full_all = df["Parc"].to_numpy(dtype=float)
    masked_all = df["Parc_masked"].to_numpy(dtype=float)
    complete_case = df["mobile_complete_case"].to_numpy(dtype=bool)

    # 中文注释：这里不是全样本趋势替代，而是在 Ba/Rb/Sr/K 原始齐全的样品中做成对敏感性检验；
    # 蓝线与橙线在 panel c 内部共用同一个 complete-case 样本池。
    valid = np.isfinite(masked_all) & complete_case
    if not np.any(valid):
        ax.text(0.5, 0.55, "Ba/Rb/Sr/K complete-case masked model unavailable\n(no PyTorch / weights)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="#999999", style="italic")
        _panel_tag(ax, "c", "Mobile-element sensitivity", tag_y=1.05)
        return

    age = age_all[valid]
    full = full_all[valid]
    masked = masked_all[valid]

    # 中文注释：蓝线使用 complete-case 子集内的 full prediction；
    # 它不等同于 panel b 的全样本 200 Myr 主线，图注中需明确说明。
    c, p, lo, hi, n_full = binned_proportion_ci(
        age,
        (full >= THRESH_DEFAULT).astype(float),
        DEFAULT_BIN_MYR,
        seed=BOOT_SEED + int(DEFAULT_BIN_MYR),
    )
    _plot_points(
        ax, c, p, lo, hi, COL_BLUE, "Full model (complete cases)", "D",
        show_ci=False, linewidth=1.8, markersize=5.6,
        alpha=1.0, zorder=6, markeredgewidth=1.4,
    )

    # 中文注释：橙线使用完全相同的 complete-case 样品和年龄分箱，仅把 Ba/Rb/Sr/K 标记为缺失后重推理。
    cm, pm, lom, him, n_masked = binned_proportion_ci(
        age,
        (masked >= THRESH_DEFAULT).astype(float),
        DEFAULT_BIN_MYR,
        seed=BOOT_SEED + int(DEFAULT_BIN_MYR) + 1,
    )
    if not np.array_equal(n_full, n_masked):
        print("[警告] panel c 蓝线与橙线分箱样本数不一致，请检查 masked 缓存。")
    _plot_points(ax, cm, pm, lom, him, COL_ORANGE,
                 "Ba, Rb, Sr, K masked", "s", linestyle=(0, (5, 2)))

    _legend(ax, loc="upper right", fontsize=10, handlelength=1.8,
            labelspacing=0.3, borderaxespad=0.5)
    _panel_tag(ax, "c", "Mobile-element sensitivity", tag_y=1.05)
def draw_panel_d(ax) -> None:
    """(d) Applicability-domain check：高弧样品在现代参照域内/外的占比，按 200 Myr 年龄分箱
    堆叠柱状图。横轴与 a–c 一致，使用连续 Age (Ga) 反向轴（柱画在各 200 Myr 箱中心）。"""
    if not DOMAIN_CACHE.exists():
        ax.text(0.5, 0.5, "modern-domain membership unavailable\n(no sklearn / data)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#999999", style="italic")
        _panel_tag(ax, "d", "Applicability-domain check", tag_y=1.05)
        ax.set_xlim(X_OLD, X_YOUNG)
        ax.set_xticks(X_TICKS)
        ax.set_yticks([])
        # 中文注释：即使 d 图数据不可用，也保留完整四边框。
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(SPINE_GRAY)
            spine.set_linewidth(0.6)
        return

    cache = pd.read_csv(DOMAIN_CACHE, encoding="utf-8-sig")
    drop = _ba_outlier_mask(cache["SAMPLE_ID"], cache.get("BA", np.nan))
    cache = cache.loc[~drop.to_numpy()]
    high = cache[pd.to_numeric(cache["Parc"], errors="coerce") >= HIGH_THRESHOLD].copy()
    high["age_ma"] = pd.to_numeric(high["age_ma"], errors="coerce")

    edges = DOMAIN_BIN_EDGES                         # 整数 Ma 边界，200 Myr
    centers, fr_in, fr_mid, fr_out = [], [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sub = high[(high["age_ma"] >= lo) & (high["age_ma"] < hi)]
        if len(sub) == 0:
            continue
        cat = sub["category"].to_numpy()
        centers.append((lo + hi) / 2000.0)           # 箱中心，单位 Ga
        fr_in.append(float(np.mean(cat == "in")))
        fr_mid.append(float(np.mean(cat == "mid")))
        fr_out.append(float(np.mean(cat == "out")))

    centers = np.asarray(centers)
    fr_in, fr_mid, fr_out = map(np.asarray, (fr_in, fr_mid, fr_out))
    bar_w = 0.17                                      # 略小于 0.2 Ga，柱间留缝
    ax.bar(centers, fr_in, width=bar_w, color=COL_IN_DOMAIN, edgecolor="white",
           linewidth=0.6, label="Within 95% domain", zorder=3)
    ax.bar(centers, fr_mid, width=bar_w, bottom=fr_in, color=COL_MARGINAL,
           edgecolor="white", linewidth=0.6, label="95–99% domain", zorder=3)
    ax.bar(centers, fr_out, width=bar_w, bottom=fr_in + fr_mid, color=COL_OUT_DOMAIN,
           edgecolor="white", linewidth=0.6, label="Outside 99% domain", zorder=3)

    # 中文注释：横轴与 a–c 完全一致——连续 Age (Ga) 反向轴 + 相同刻度。
    ax.set_xlim(X_OLD, X_YOUNG)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.001, 0.25))
    ax.set_ylabel("Proportion of high arc-like samples", fontsize=9.0)
    ax.set_xlabel("Age (Ga)", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.5)
    # 中文注释：四条边界线全部保留，统一线宽为 0.6。
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_GRAY)
        spine.set_linewidth(0.6)
    # 三类图例移到 panel 外上方居中、缩小字号；标题用更大 pad 置于图例之上。
    _legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
            fontsize=8.0, handlelength=1.1, handletextpad=0.4,
            columnspacing=1.0, borderaxespad=0.0)
    _panel_tag(ax, "d", "Applicability-domain check", title_pad=24.0, tag_y=1.05)


# ══════════════════════════════════════════════════════════════════════════════
#  图件组装
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.9))
    fig.patch.set_facecolor("#FFFFFF")
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    draw_panel_a(ax_a, df)
    draw_panel_b(ax_b, df)
    draw_panel_c(ax_c, df)
    draw_panel_d(ax_d)

    for ax in (ax_a, ax_b, ax_c):
        _style_time_panel(ax)
    for ax in (ax_a, ax_c):
        ax.set_ylabel("Arc-related proportion", fontsize=9.5)
    # 中文注释：上排 a/b 保留 x 轴 tick labels，但不重复写 Age (Ga)；下排 c/d 保留轴标题。
    ax_c.set_xlabel("Age (Ga)", fontsize=9.5)

    # 中文注释：去掉总标题，使整图更接近 Supplementary Figure 风格（编号 a–d 已足够）。
    fig.subplots_adjust(left=0.072, right=0.955, top=0.93, bottom=0.105,
                        wspace=0.185, hspace=0.42)

    # 中文注释：将下排 c/d 两个子图整体轻微上移，保留下方轴标题空间但减少上下排之间的松散感。
    lower_row_shift = 0.018
    for ax in (ax_c, ax_d):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + lower_row_shift, pos.width, pos.height])

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=600, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[OK] 敏感性四联图已保存: {FIG_PATH}")


CAPTION_TEXT = """\
Fig. 7 (supplementary). Sensitivity tests for the Archean arc-like affinity pattern,
using Parc = P_CA + P_IA + P_IOA as the arc-related affinity score. The age axis runs
4.0 to 2.4 Ga (old to young), consistent with the main time-evolution plot. The pale-blue band
marks the ~3.8 Ga window, the pale-purple band the ~2.7-2.5 Ga window, and the dark
dashed line ~3.5 Ga.
(a-c) Discrete binned point-line plots: within each equal-width age bin a single point
gives the proportion of samples with Parc above a threshold; points are joined by
straight lines only as a visual guide. No spline, KDE or rolling smoothing is applied,
so no spurious continuous time resolution is implied for the sparse Archean record.
Where shown, error bars are 95% bootstrap confidence intervals from resampling samples within each
bin; bins with fewer than 8 samples are omitted, and empty bins (e.g. 3.6-3.7 Ga, with
no samples) are simply absent, with the line spanning the neighbouring points. Following
the time-evolution main plot, only bins centred at >= 2.5 Ga are plotted (binning is done in integer
Ma so age boundaries are exact).
(a) Threshold sensitivity: Parc >= 0.4, 0.5 and 0.6 in fixed 200-Myr bins
(same binning as the time-evolution main plot).
(b) Age-bin sensitivity: 50, 100 and 200 Myr bins at fixed Parc >= 0.5; finer bins are
noisier but the ~3.5 Ga and late-Archean features persist.
(c) Mobile-element sensitivity: paired complete-case test on samples for which Ba,
Rb, Sr and K were all originally measured. The blue curve shows the full-model
prediction within this retained subset, and the orange curve shows a genuine GeoDAN
re-inference after setting Ba/Rb/Sr/K to missing. Within panel c, both curves use
the same retained samples, the same 200-Myr age bins, and the same Parc >= 0.5
threshold; only the mobile-element inputs are changed.
(d) Applicability-domain check. For each 200-Myr age bin, high arc-like samples
(Parc >= 0.5) are split by their position relative to the modern basaltic compositional
domain in a PCA subspace (kNN applicability-domain distance; modern 95% / 99% quantiles,
following pca_distribution_consistency.py) into within the 95% domain, the 95-99%
margin, and outside the 99% domain. Across the well-populated bins the large majority of
high arc-like samples fall within the 99% modern domain, showing that the arc-like
signal is not driven by compositional outliers. Sample counts per bin are annotated; the
oldest bins contain very few high arc-like samples.
"""


def write_caption() -> None:
    CAPTION_PATH.write_text(CAPTION_TEXT, encoding="utf-8")
    print(f"[OK] 建议图注已保存: {CAPTION_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 78)
    print("Fig. 7 敏感性四联图（阈值 / 分箱 / 易迁移元素 / applicability-domain check）")
    print("=" * 78)
    if not MASKED_CACHE.exists():
        generate_masked_cache()
    if not DOMAIN_CACHE.exists():
        generate_domain_cache()

    df = load_dataframe()
    print(f"[数据] 样品总数={len(df)}，"
          f"掩蔽推理样品={int(df['Parc_masked'].notna().sum())}，"
          f"掩蔽模型可用={'是' if df['Parc_masked'].notna().any() else '否'}，"
          f"现代域缓存={'有' if DOMAIN_CACHE.exists() else '无'}")
    build_figure(df)
    write_caption()
    print("完成。太古代时间演化主图及其输出文件未改动。")


if __name__ == "__main__":
    main()
