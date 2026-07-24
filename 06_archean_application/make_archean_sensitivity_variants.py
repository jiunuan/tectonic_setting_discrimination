from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

# 中文注释：先把当前项目根目录加入导入路径，再读取集中路径配置。

# 中文注释：直接写完整工程路径，避免相对路径在 IDE/命令行切换时失效。
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ARCHEAN_FINAL_DIR, ARCHEAN_FIG7_SENSITIVITY_DIR

import archean_vit_transformer_dualstream_predict_analysis as A  # noqa: E402
import archean_time_evolution_sensitivity_four_panel as F  # noqa: E402

VARIANT_DIR = ARCHEAN_FIG7_SENSITIVITY_DIR
SUMMARY_CSV = VARIANT_DIR / "panel_c_variant_summary_same_n.csv"

# 中文注释：归一化特征列到原始表列的映射，用于 complete-case 子集审计。
RAW_COLUMN_MAP = {
    "BA(PPM)": "BA",
    "RB(PPM)": "RB",
    "SR(PPM)": "SR",
    "K2O(WT%)": "K2O",
}

# 中文注释：每个候选方案都使用“值置 0 + missing_mask 置 1”的显式缺失编码重推理。
ELEMENT_SPECS = [
    {
        "short": "BaRbSrK",
        "label": "Ba, Rb, Sr, K masked",
        "columns": ["BA(PPM)", "RB(PPM)", "SR(PPM)", "K2O(WT%)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_BaRbSrK_predictions.csv",
    },    {
        "short": "Ba_only",
        "label": "Ba masked",
        "columns": ["BA(PPM)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_Ba_predictions.csv",
    },    {
        "short": "BaK",
        "label": "Ba, K masked",
        "columns": ["BA(PPM)", "K2O(WT%)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_BaK_predictions.csv",
    },
    {
        "short": "BaRbK",
        "label": "Ba, Rb, K masked",
        "columns": ["BA(PPM)", "RB(PPM)", "K2O(WT%)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_BaRbK_predictions.csv",
    },
    {
        "short": "BaSrK",
        "label": "Ba, Sr, K masked",
        "columns": ["BA(PPM)", "SR(PPM)", "K2O(WT%)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_BaSrK_predictions.csv",
    },
    {
        "short": "BaRbSr_no_K",
        "label": "Ba, Rb, Sr masked",
        "columns": ["BA(PPM)", "RB(PPM)", "SR(PPM)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_BaRbSr_predictions.csv",
    },
    {
        "short": "K_only",
        "label": "K masked",
        "columns": ["K2O(WT%)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_K_predictions.csv",
    },
    {
        "short": "Sr_only",
        "label": "Sr masked",
        "columns": ["SR(PPM)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_Sr_predictions.csv",
    },
    {
        "short": "BaRb_only",
        "label": "Ba, Rb masked",
        "columns": ["BA(PPM)", "RB(PPM)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_BaRb_predictions.csv",
    },
    {
        "short": "RbSrK_no_Ba",
        "label": "Rb, Sr, K masked",
        "columns": ["RB(PPM)", "SR(PPM)", "K2O(WT%)"],
        "cache": ARCHEAN_FINAL_DIR / "fig7_sensitivity_masked_RbSrK_predictions.csv",
    },
]

SAMPLE_POOLS = [
    {
        "name": "all_samples",
        "title_suffix": "all samples",
        "output_suffix": "all_samples_200M",
        "complete_case": False,
    },
    {
        "name": "complete_case_for_masked_elements",
        "title_suffix": "complete-case",
        "output_suffix": "complete_case_200M",
        "complete_case": True,
    },
]


def _drop_outlier_mask() -> np.ndarray:
    """中文注释：复用主图同一条 Ba 异常样品剔除规则，保证缓存与 df 行严格对齐。"""
    raw = F._read_prediction_table(F.PREDICTION_CSV)
    return F._ba_outlier_mask(
        raw.get("SAMPLE_ID", pd.Series("", index=raw.index)).astype(str),
        raw.get("BA", np.nan),
    ).to_numpy()


def _ensure_masked_cache(cache_path: Path, columns: list[str], label: str) -> None:
    """中文注释：为指定元素集合生成“全样本显式缺失编码”的 GeoDAN 重推理缓存。"""
    if cache_path.exists():
        return
    if not getattr(A, "_TORCH_AVAILABLE", False):
        raise RuntimeError(f"当前环境无 PyTorch，无法生成 {label} 缓存")

    class_names = A.load_class_names(A.TRAIN_PATH)
    metadata = A.load_final_age_constrained_pool()
    metadata, normalized = A._prepare_archean_features_from_metadata(metadata)
    missing_mask = A._build_archean_missing_mask(metadata)
    normalized, missing_mask = normalized.copy(), missing_mask.copy()

    for column in columns:
        if column not in normalized.columns:
            raise KeyError(f"归一化特征缺少列: {column}")
        mask_column = f"missing_mask__{column}"
        if mask_column not in missing_mask.columns:
            raise KeyError(f"缺失掩码缺少列: {mask_column}")
        normalized[column] = 0
        missing_mask[mask_column] = 1

    probs = A._predict_with_missing_mask(
        normalized,
        missing_mask,
        class_names,
        A.FINAL_MODEL_WEIGHT_PATH,
    )
    pred = A.add_prediction_columns(
        metadata,
        probs,
        np.zeros_like(probs),
        class_names,
        high_prob=A.HIGH_PROB,
        high_std=A.HIGH_STD,
    )
    pd.DataFrame({
        "SAMPLE_ID": metadata["SAMPLE_ID"].astype(str).to_numpy(),
        "Parc_masked": pred["Arc_probability3"].to_numpy(),
    }).to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已生成 {label} 缓存: {cache_path}")


def _load_masked_cache(cache_path: Path, drop_mask: np.ndarray) -> np.ndarray:
    """中文注释：按原始行序读取缓存，再应用同一个异常样品剔除 mask；不按 SAMPLE_ID 对齐，因为存在重复 ID。"""
    cache = pd.read_csv(cache_path, encoding="utf-8-sig")
    return pd.to_numeric(cache["Parc_masked"], errors="coerce").to_numpy()[~drop_mask]


def _complete_case_filter(columns: list[str]) -> np.ndarray:
    """中文注释：只在输出 complete-case 方案时使用；图内蓝/橙仍共用这一样本池。"""
    raw = F._read_prediction_table(F.PREDICTION_CSV)
    drop_mask = _drop_outlier_mask()
    raw_columns = [RAW_COLUMN_MAP[column] for column in columns]
    missing_columns = [column for column in raw_columns if column not in raw.columns]
    if missing_columns:
        raise KeyError(f"原始表缺少 complete-case 列: {missing_columns}")
    complete = raw[raw_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    return complete[~drop_mask]


def _build_sample_filter(pool: dict[str, object], columns: list[str], masked: np.ndarray) -> np.ndarray:
    """中文注释：构建样本池；所有方案都额外要求 masked 推理结果有效。"""
    valid = np.isfinite(masked)
    if bool(pool["complete_case"]):
        valid = valid & _complete_case_filter(columns)
    return valid


def _same_n_panel_c(
    ax,
    df: pd.DataFrame,
    masked: np.ndarray,
    sample_filter: np.ndarray,
    label: str,
    pool_name: str,
) -> list[dict[str, object]]:
    """中文注释：绘制严格同样本池、同 200 Myr 分箱的 panel c，并返回每个分箱的数值。"""
    F._draw_windows(ax)
    age_all = df["age_ga"].to_numpy(dtype=float)
    full_all = df["Parc"].to_numpy(dtype=float)
    age = age_all[sample_filter]
    full = full_all[sample_filter]
    masked_valid = masked[sample_filter]

    centers, props, lows, highs, n_full = F.binned_proportion_ci(
        age,
        (full >= F.THRESH_DEFAULT).astype(float),
        F.DEFAULT_BIN_MYR,
        seed=F.BOOT_SEED + int(F.DEFAULT_BIN_MYR),
    )
    centers_m, props_m, lows_m, highs_m, n_masked = F.binned_proportion_ci(
        age,
        (masked_valid >= F.THRESH_DEFAULT).astype(float),
        F.DEFAULT_BIN_MYR,
        seed=F.BOOT_SEED + int(F.DEFAULT_BIN_MYR) + 1,
    )

    full_label = "Full model (200 Myr)" if pool_name == "all_samples" else "Full model (same samples)"
    F._plot_points(
        ax, centers, props, lows, highs, F.COL_BLUE,
        full_label, "D", show_ci=False,
        linewidth=1.8, markersize=5.6, alpha=1.0,
        zorder=6, markeredgewidth=1.4,
    )
    F._plot_points(
        ax, centers_m, props_m, lows_m, highs_m, F.COL_ORANGE,
        label, "s", linestyle=(0, (5, 2)),
    )
    F._legend(ax, loc="upper right", fontsize=9.5, handlelength=1.8,
              labelspacing=0.3, borderaxespad=0.5)
    F._panel_tag(ax, "c", "Mobile-element sensitivity", tag_y=1.05)

    center_to_masked = {
        round(float(center), 6): (float(prop), int(n))
        for center, prop, n in zip(centers_m, props_m, n_masked)
    }
    rows = []
    for center, full_prop, full_n in zip(centers, props, n_full):
        masked_prop, masked_n = center_to_masked.get(round(float(center), 6), (np.nan, -1))
        rows.append({
            "age_ga": float(center),
            "n_full": int(full_n),
            "n_masked": int(masked_n),
            "same_n": bool(int(full_n) == int(masked_n)),
            "full_high": float(full_prop),
            "masked_high": float(masked_prop),
            "diff_masked_minus_full": float(masked_prop - full_prop),
        })
    return rows


def _build_variant(
    df: pd.DataFrame,
    element_spec: dict[str, object],
    pool: dict[str, object],
    masked: np.ndarray,
) -> list[dict[str, object]]:
    """中文注释：临时替换主脚本 panel c，输出完整四联图候选版本。"""
    sample_filter = _build_sample_filter(pool, list(element_spec["columns"]), masked)
    rows_holder: list[dict[str, object]] = []
    output_path = VARIANT_DIR / f"fig7_variant_{element_spec['short']}_{pool['output_suffix']}.png"

    old_draw_panel_c = F.draw_panel_c
    old_fig_path = F.FIG_PATH
    try:
        def _patched_panel_c(ax, current_df):
            rows_holder[:] = _same_n_panel_c(
                ax,
                current_df,
                masked,
                sample_filter,
                str(element_spec["label"]),
                str(pool["name"]),
            )

        F.draw_panel_c = _patched_panel_c
        F.FIG_PATH = output_path
        F.build_figure(df)
    finally:
        F.draw_panel_c = old_draw_panel_c
        F.FIG_PATH = old_fig_path

    raw_columns = [RAW_COLUMN_MAP[column] for column in element_spec["columns"]]
    for row in rows_holder:
        row["variant"] = f"{element_spec['short']}_{pool['name']}"
        row["masked_elements"] = ", ".join(element_spec["columns"])
        row["raw_complete_case_columns"] = ", ".join(raw_columns) if bool(pool["complete_case"]) else "N/A"
        row["sample_pool"] = str(pool["name"])
        row["sample_pool_n_total"] = int(sample_filter.sum())
        row["figure_path"] = str(output_path)
    return rows_holder


def _load_all_sample_variant_series(summary: pd.DataFrame) -> pd.DataFrame:
    """中文注释：读取全样本方案的年龄-比例矩阵，用于绘制敏感性包络和差值热图。"""
    all_samples = summary[summary["sample_pool"].eq("all_samples")].copy()
    if all_samples.empty:
        raise ValueError("缺少 all_samples 方案，无法绘制敏感性汇总图")
    if not bool(all_samples["same_n"].all()):
        raise ValueError("all_samples 方案存在 n 不一致，不能绘制敏感性汇总图")
    return all_samples


def _draw_panel_c_envelope(ax, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """中文注释：候选 panel c：蓝线为原模型，橙色带为多种缺失编码方案的范围。"""
    F._draw_windows(ax)
    all_samples = _load_all_sample_variant_series(summary)
    age = all_samples["age_ga"].drop_duplicates().sort_values().to_numpy(dtype=float)
    full = all_samples.drop_duplicates("age_ga").sort_values("age_ga")["full_high"].to_numpy(dtype=float)
    grouped = all_samples.groupby("age_ga")["masked_high"]
    low = grouped.min().reindex(age).to_numpy(dtype=float)
    high = grouped.max().reindex(age).to_numpy(dtype=float)
    median = grouped.median().reindex(age).to_numpy(dtype=float)

    zeros = np.zeros_like(full)
    F._plot_points(
        ax, age, full, zeros, zeros, F.COL_BLUE,
        "Full model (200 Myr)", "D", show_ci=False,
        linewidth=1.8, markersize=5.6, alpha=1.0,
        zorder=6, markeredgewidth=1.4,
    )
    ax.fill_between(age, low, high, color=F.COL_ORANGE, alpha=0.18,
                    linewidth=0, label="Masked-element range")
    F._plot_points(
        ax, age, median, zeros, zeros, F.COL_ORANGE,
        "Masked median", "s", show_ci=False,
        linestyle=(0, (5, 2)), linewidth=1.5, markersize=4.6,
        alpha=1.0, zorder=5,
    )
    F._legend(ax, loc="upper right", fontsize=9.5, handlelength=1.8,
              labelspacing=0.3, borderaxespad=0.5)
    F._panel_tag(ax, "c", "Mobile-element sensitivity", tag_y=1.05)


def _draw_panel_c_delta_heatmap(ax, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """中文注释：候选 panel c：用差值热图展示各缺失编码方案的影响幅度。"""
    all_samples = _load_all_sample_variant_series(summary)
    order = [
        "K_only_all_samples",
        "Sr_only_all_samples",
        "BaRb_only_all_samples",
        "RbSrK_no_Ba_all_samples",
        "BaRbSrK_all_samples",
        "BaRbSr_no_K_all_samples",
    ]
    labels = [
        "K",
        "Sr",
        "Ba+Rb",
        "Rb+Sr+K",
        "Ba+Rb+Sr+K",
        "Ba+Rb+Sr",
    ]
    ages = sorted(all_samples["age_ga"].unique())
    matrix = []
    for variant in order:
        sub = all_samples[all_samples["variant"].eq(variant)].set_index("age_ga")
        matrix.append(sub.reindex(ages)["diff_masked_minus_full"].to_numpy(dtype=float))
    matrix = np.asarray(matrix, dtype=float)

    vmax = max(0.18, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(ages)))
    ax.set_xticklabels([f"{age:.1f}" for age in ages], fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Age (Ga)")
    ax.set_ylabel("Masked elements")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    fontsize=7.2, color="white" if abs(value) > vmax * 0.45 else "black")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ arc proportion", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.text(0.0, 1.04, "c", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="left")
    ax.set_title("Mobile-element sensitivity", fontsize=11, pad=6)


def _build_summary_panel_variant(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    panel_kind: str,
) -> None:
    """中文注释：临时替换 panel c，输出敏感性汇总型四联图。"""
    old_draw_panel_c = F.draw_panel_c
    old_fig_path = F.FIG_PATH
    try:
        if panel_kind == "envelope":
            def _patched_panel_c(ax, current_df):
                _draw_panel_c_envelope(ax, current_df, summary)
        elif panel_kind == "delta_heatmap":
            def _patched_panel_c(ax, current_df):
                _draw_panel_c_delta_heatmap(ax, current_df, summary)
        else:
            raise ValueError(f"未知 panel_kind: {panel_kind}")
        F.draw_panel_c = _patched_panel_c
        F.FIG_PATH = output_path
        F.build_figure(df)
    finally:
        F.draw_panel_c = old_draw_panel_c
        F.FIG_PATH = old_fig_path

def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    for element_spec in ELEMENT_SPECS:
        _ensure_masked_cache(
            Path(str(element_spec["cache"])),
            list(element_spec["columns"]),
            str(element_spec["short"]),
        )

    drop_mask = _drop_outlier_mask()
    df = F.load_dataframe()

    rows = []
    for element_spec in ELEMENT_SPECS:
        masked = _load_masked_cache(Path(str(element_spec["cache"])), drop_mask)
        if len(masked) != len(df):
            raise ValueError(f"{element_spec['short']} 缓存长度与主表不一致: {len(masked)} != {len(df)}")
        for pool in SAMPLE_POOLS:
            rows.extend(_build_variant(df, element_spec, pool, masked))
            print(f"[OK] 已输出方案: {element_spec['short']} | {pool['name']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 方案数值汇总已保存: {SUMMARY_CSV}")

    envelope_path = VARIANT_DIR / "fig7_variant_G_all_samples_masking_envelope.png"
    heatmap_path = VARIANT_DIR / "fig7_variant_H_all_samples_delta_heatmap.png"
    _build_summary_panel_variant(df, summary, envelope_path, "envelope")
    _build_summary_panel_variant(df, summary, heatmap_path, "delta_heatmap")
    print(f"[OK] 已输出敏感性包络图: {envelope_path}")
    print(f"[OK] 已输出差值热图: {heatmap_path}")
    print(summary[[
        "variant", "age_ga", "n_full", "n_masked", "same_n",
        "full_high", "masked_high", "diff_masked_minus_full",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
