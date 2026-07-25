"""集中路径配置（Centralised path configuration）
================================================================
本项目所有脚本通过本模块获取数据 / 模型 / 输出路径，
不再使用硬编码绝对路径，从而保证跨机器、跨平台可移植。

用法（在每个脚本顶部加入）::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.paths import NORMALIZED_DIR, MODELS_DIR, TRAIN_NORM_CSV, ...

约定
----
- 所有路径均相对 ``PROJECT_ROOT`` 推导，clone 仓库后即可运行。
- 大数据与模型权重默认放在 ``data/``（已在 .gitignore 中排除）。
- 历史遗留的 ``dataset_split_correct`` / ``06_normalize_*`` 命名一律
  统一到本模块的 ``06_normalized`` / ``05_normalize_*`` 基准。
"""

import os
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# 项目根目录（本文件位于 <root>/config/paths.py）
# ════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 数据根 ───────────────────────────────────────────────────────
# 默认使用仓库内的相对 data 目录；Code Ocean 可通过环境变量把它改到可写的 scratch。
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(os.environ.get("BASALT_DATA_DIR", str(DEFAULT_DATA_DIR)))
# Zenodo 精简发布数据集（三张公开共享表；不纳入 GitHub）。
ZENODO_DATASET_DIR = DATA_DIR / "basalt_geochemistry_dataset"
ZENODO_MODERN_CSV = ZENODO_DATASET_DIR / "modern_basalt_geochemistry.csv"
ZENODO_ARCHEAN_CSV = ZENODO_DATASET_DIR / "archean_basalt_geochemistry.csv"
ZENODO_ARCHEAN_PREDICTIONS_CSV = ZENODO_DATASET_DIR / "archean_basalt_geodan_predictions.csv"


def _first_existing_path(candidates):
    """中文注释：优先使用标准文件名，同时兼容历史版本留下的文件名。"""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

# 各阶段目录（与 README 流程图编号一致）
RAW_DIR        = DATA_DIR / "00_raw"            # 原始数据
CM_RECLASS_DIR = DATA_DIR / "01_cm_reclass_input"  # 汇聚边缘细分产出（输入起点）
FILTERED_DIR   = DATA_DIR / "02_filtered"       # 筛选后
COMBINED_DIR   = DATA_DIR / "03_combined"       # GEOROC + PetDB 合并
SPLIT_DIR      = DATA_DIR / "04_split"          # 训练/测试切分
IMPUTED_DIR    = DATA_DIR / "05_imputed"        # 全局 RF 插补 + 缺失 mask
NORMALIZED_DIR = DATA_DIR / "06_normalized"     # 主量无水标准化 + SMOTE + 分位归一化
MODELS_DIR     = DATA_DIR / "models"            # 训练产出权重 .pth
ARCHEAN_DIR    = DATA_DIR / "archean"           # 太古代应用数据 + 案例 + 输出

# ════════════════════════════════════════════════════════════════
# 00_raw —— 原始数据文件
# ════════════════════════════════════════════════════════════════
GEOROC_RAW_CSV        = RAW_DIR / "georoc" / "basalt_2025.csv"
PETDB_RAW_CSV = _first_existing_path(
    [
        RAW_DIR / "petdb" / "petdbv2_merged.csv",
        RAW_DIR / "petdb" / "petDB_recent_downloads_merged.csv",
    ]
)  # PetDB 2.0 合并原始表
GEOROC_REFERENCES_CSV = _first_existing_path(
    [
        RAW_DIR / "georoc" / "references_structured.csv",
        CM_RECLASS_DIR / "references_structured.csv",
    ]
)  # GEOROC 参考文献编号→年份映射

# ════════════════════════════════════════════════════════════════
# 01_cm_reclass_input —— 汇聚边缘细分项目（convergent_margin_reclass）的产出
# 本项目不含该细分代码，仅以其产出作为输入起点
# ════════════════════════════════════════════════════════════════
CM_CORE_CSV          = CM_RECLASS_DIR / "georoc_convergent_margin_core_training_high_confidence.csv"
CM_EXPANDED_CSV      = CM_RECLASS_DIR / "georoc_convergent_margin_expanded_training_reviewed.csv"
REFINED_EXPANDED_CSV = CM_RECLASS_DIR / "basalt_refined_expanded.csv"  # 用细分结果重分类后的 GEOROC

# ════════════════════════════════════════════════════════════════
# 02_filtered —— 经筛选规则后的数据
# ════════════════════════════════════════════════════════════════
GEOROC_FILTERED_CSV = _first_existing_path(
    [
        FILTERED_DIR / "basalt_refined_expanded_filtered.csv",
        FILTERED_DIR / "basalt_refined_expanded_whole_rock_filtered.csv",
    ]
)
PETDB_FILTERED_CSV = _first_existing_path(
    [
        FILTERED_DIR / "petDB.csv",
        FILTERED_DIR / "petdbv2_merged_filtered.csv",
    ]
)

# ════════════════════════════════════════════════════════════════
# 03_combined —— GEOROC + PetDB 合并
# ════════════════════════════════════════════════════════════════
COMBINED_CSV = _first_existing_path(
    [
        COMBINED_DIR / "01_basalt_number_year.csv",
        COMBINED_DIR / "01_basalt_number_year_wr.csv",
    ]
)

# ════════════════════════════════════════════════════════════════
# 04_split —— 训练/测试切分
# ════════════════════════════════════════════════════════════════
TRAIN_RAW_CSV      = SPLIT_DIR / "01_basalt_number_year_train.csv"
TEST_RAW_CSV       = SPLIT_DIR / "01_basalt_number_year_test.csv"
SPLIT_SUMMARY_CSV  = SPLIT_DIR / "split_summary.csv"

# ════════════════════════════════════════════════════════════════
# 05_imputed —— 全局随机森林插补（训练集 fit / 测试集 transform）
#               + 插补前的原始缺失 mask（1=原始缺失，0=原始实测）
# ════════════════════════════════════════════════════════════════
TRAIN_IMPUTED_CSV = IMPUTED_DIR / "02_basalt_train_imputed.csv"
TEST_IMPUTED_CSV  = IMPUTED_DIR / "02_basalt_test_imputed.csv"
MASK_TRAIN_CSV    = IMPUTED_DIR / "03_train_missing_mask.csv"
MASK_TEST_CSV     = IMPUTED_DIR / "03_test_missing_mask.csv"

# ════════════════════════════════════════════════════════════════
# 06_normalized —— 主量无水标准化 + 选择性 SMOTE + 分位数分箱归一化
# ════════════════════════════════════════════════════════════════
TRAIN_MAJOR_NORM_CSV    = NORMALIZED_DIR / "04_basalt_train_major_normalize.csv"
TEST_MAJOR_NORM_CSV     = NORMALIZED_DIR / "04_basalt_test_major_normalize.csv"
TRAIN_SMOTE_CSV         = NORMALIZED_DIR / "05_basalt_train_selected_smote.csv"   # 仅训练集执行 SMOTE
TRAIN_NORM_CSV          = NORMALIZED_DIR / "06_normalize_basalt_train.csv"        # 最终喂模型/SHAP 的训练集（SMOTE 后）
TRAIN_NORM_NO_SMOTE_CSV = NORMALIZED_DIR / "06_normalize_basalt_train_no_smote.csv"  # 未 SMOTE 的真实训练集（对照实验用）
TEST_NORM_CSV           = NORMALIZED_DIR / "06_normalize_basalt_test.csv"         # 最终喂模型/SHAP 的测试集
QUANTILE_PARAMS_JSON    = NORMALIZED_DIR / "quantile_params.json"                 # 分位参数（从 SMOTE 前训练集拟合）

# ════════════════════════════════════════════════════════════════
# models —— 训练产出权重
# ════════════════════════════════════════════════════════════════
GEODAN_DIR        = MODELS_DIR / "GeoDAN"
MAIN_MODEL_WEIGHT = MODELS_DIR / "Full_Model_(ViT+Transformer)_best_seed.pth"
# GeoDAN 训练后重绘 ROC/PR 曲线使用的缓存和输出。
ROC_PR_CURVE_DATA_CSV = MODELS_DIR / "roc_pr_sci_comparison_curve_data.csv"
ROC_PR_FROM_SAVED_PNG = MODELS_DIR / "roc_pr_sci_comparison_from_saved_curve_data.png"

# SHAP 面板缓存与重绘输出。
SHAP_ANALYSIS_DIR = MODELS_DIR / "shap_analysis"
SHAP_FIGURE7_PANEL_DIR = SHAP_ANALYSIS_DIR / "figure7_panels_true_class_median"
SHAP_FIGURE7A_PATH = SHAP_FIGURE7_PANEL_DIR / "Figure7a_heatmap.png"
SHAP_FIGURE7C_PATH = SHAP_FIGURE7_PANEL_DIR / "Figure7c_ranking.png"
SHAP_FIGURE7AC_PATH = SHAP_FIGURE7_PANEL_DIR / "Figure7a_c_combined.png"

# ════════════════════════════════════════════════════════════════
# archean —— 太古代应用（缺失编码：不插补，缺失值数值编码 0 + mask 1）
# ════════════════════════════════════════════════════════════════
ARCHEAN_DATA_SUBDIR = ARCHEAN_DIR / "data"          # 太古代原始 CSV（Liu 数据）+ 6 克拉通案例
ARCHEAN_S3_CSV      = ARCHEAN_DATA_SUBDIR / "archean_basalt.csv"
ARCHEAN_OUTPUT_DIR  = ARCHEAN_DIR / "outputs"        # 太古代预处理 / 预测输出

# 扩展太古代应用集（Liu SiO2≤54 放宽 + GeoROC 恢复的 ARCHEAN 样品）
ARCHEAN_POOL_DIR        = ARCHEAN_OUTPUT_DIR / "extended_archean_pool"
ARCHEAN_POOL_RAW_CSV    = ARCHEAN_POOL_DIR / "expanded_archean_raw.csv"
ARCHEAN_POOL_CSV = _first_existing_path(
    [
        ARCHEAN_POOL_DIR / "expanded_archean_basalt_age_nonmissing.csv",
        ARCHEAN_DATA_SUBDIR / "expanded_archean_basalt_age_nonmissing.csv",
        ZENODO_ARCHEAN_CSV,
    ]
)  # 候选池 3,483 条(SiO2≤54)；按 SiO2≤53 筛得正式应用集 3,012 条
ARCHEAN_POOL_MASK_CSV   = ARCHEAN_POOL_DIR / "expanded_archean_missing_mask.csv"

# 正式缺失编码预测输出（GeoDAN final）
ARCHEAN_FINAL_DIR             = ARCHEAN_OUTPUT_DIR / "archean_geodan_final"
ARCHEAN_FINAL_MASK_CSV        = ARCHEAN_FINAL_DIR / "expanded_archean_missing_mask.csv"
ARCHEAN_FINAL_PREDICTIONS_CSV = ARCHEAN_FINAL_DIR / "expanded_archean_predictions.csv"
ARCHEAN_FIG7_SENSITIVITY_DIR  = ARCHEAN_FINAL_DIR / "fig7_sensitivity_variants"
# 6 克拉通案例研究输出
ARCHEAN_CASE_DIR = ARCHEAN_OUTPUT_DIR / "archean_case_studies"
ARCHEAN_CASE_PREDICTIONS_DIR = ARCHEAN_CASE_DIR / "predictions"
# 分布一致性 / 适用域诊断输出
ARCHEAN_CONSISTENCY_DIR = ARCHEAN_OUTPUT_DIR / "distribution_consistency"

# ════════════════════════════════════════════════════════════════
# 论文图件输出
# ════════════════════════════════════════════════════════════════
FIGURES_DIR = DATA_DIR / "figures"
FIGURE_ASSETS_DIR = FIGURES_DIR / "assets"
WORLD_BASEMAP_PNG = FIGURE_ASSETS_DIR / "ocean_world_4326_z3_4096x1935.png"
MODERN_DISTRIBUTION_MAP_PNG = FIGURES_DIR / "distribution_basalt_map_esri.png"
ARCHEAN_DISTRIBUTION_MAP_PNG = FIGURES_DIR / "archean_basalt_geodan_prediction_distribution_map_esri.png"

# 插补方法对比附图输出。
IMPUTATION_COMPARISON_DIR = IMPUTED_DIR / "imputation_comparison_output"
