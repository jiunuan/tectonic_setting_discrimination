"""
扩展太古代玄武岩应用集并重新计算GeoDAN高弧样品数。

扩展来源：
1. Liu应用集将无水SiO2上限由53放宽到54 wt%；
2. 恢复GeoROC提取流程中因AGE包含ARCHEAN而被排除的样品；
3. 检查PetDB中同一年龄文本规则实际排除的样品数。

所有输出写入 data/archean/outputs/extended_archean_pool/。
本脚本产出 expanded_archean_raw.csv（合并去重后的扩展池）；
正式应用集需在候选池基础上使用已人工核验的地理名称和年龄结果，
并删除年龄仍为空的记录；本脚本不执行现代插补、分位数编码或模型预测。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 中文注释：全部路径集中在 config/paths.py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config.paths import (
    REFINED_EXPANDED_CSV, PETDB_RAW_CSV,
    ARCHEAN_S3_CSV,
    TRAIN_RAW_CSV, TRAIN_MAJOR_NORM_CSV, TRAIN_NORM_CSV,
    QUANTILE_PARAMS_JSON,
    ARCHEAN_POOL_DIR, ARCHEAN_POOL_RAW_CSV,
)

GEOROC_RAW_PATH = str(REFINED_EXPANDED_CSV)
PETDB_RAW_PATH = str(PETDB_RAW_CSV)
LIU_RAW_PATH = str(ARCHEAN_S3_CSV)
MODERN_TRAIN_RAW_PATH = str(TRAIN_RAW_CSV)
MODERN_TRAIN_CONTINUOUS_PATH = str(TRAIN_MAJOR_NORM_CSV)
MODEL_TRAIN_PATH = str(TRAIN_NORM_CSV)
QUANTILE_PARAMS_PATH = str(QUANTILE_PARAMS_JSON)

EXPANDED_RAW_PATH = str(ARCHEAN_POOL_RAW_CSV)
EXPANDED_IMPUTED_PATH = str(ARCHEAN_POOL_DIR / "expanded_archean_imputed_major_normalized.csv")
EXPANDED_FEATURE_PATH = str(ARCHEAN_POOL_DIR / "expanded_archean_quantile_1_255.csv")
EXPANDED_PREDICTION_PATH = str(ARCHEAN_POOL_DIR / "expanded_archean_predictions.csv")
EXPANDED_SUMMARY_PATH = str(ARCHEAN_POOL_DIR / "expanded_archean_summary.csv")
EXPANDED_REPORT_PATH = str(ARCHEAN_POOL_DIR / "expanded_archean_report.md")

SHORT_COLUMNS = [
    "NA2O", "MGO", "AL2O3", "SIO2", "P2O5", "K2O", "CAO", "TIO2", "MNO", "FEOT",
    "RB", "V", "CR", "CO", "NI", "BA", "SR", "Y", "ZR", "NB", "LA", "CE", "PR",
    "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "YB", "LU", "HF", "TA", "TH",
]
LONG_COLUMNS = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)",
    "K2O(WT%)", "CAO(WT%)", "TIO2(WT%)", "MNO(WT%)", "FEOT(WT%)",
    "RB(PPM)", "V(PPM)", "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)",
    "SR(PPM)", "Y(PPM)", "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)",
    "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)", "GD(PPM)", "TB(PPM)",
    "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)", "HF(PPM)",
    "TA(PPM)", "TH(PPM)",
]
MAJOR_SHORT = ["SIO2", "TIO2", "AL2O3", "FEOT", "MNO", "MGO", "CAO", "NA2O", "K2O", "P2O5"]
REQUIRED_MAJOR_SHORT = ["SIO2", "AL2O3", "FEOT", "MGO", "CAO"]
ARC_CLASSES = {"Continental arc", "Island arc", "Intra-oceanic arc"}
MAX_MISSING_EXCLUSIVE = 18
SIO2_MIN = 44.0
SIO2_MAX = 54.0
MGO_MAX = 18.0


def ensure_import_paths() -> None:
    """加入项目、提取脚本及本目录，复用现有预处理函数。"""
    for path in [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "01_preprocessing" / "filter"),
        str(PROJECT_ROOT / "06_archean_application"),
    ]:
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_module_from_file(module_name: str, file_path: Path):
    """中文注释：编号目录无法用常规包导入，统一用 importlib 加载。"""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_major_elements(features: pd.DataFrame) -> pd.DataFrame:
    """使用正值主量元素执行逐行无水标准化。"""
    result = features.copy()
    positive = result[MAJOR_SHORT].where(result[MAJOR_SHORT] > 0)
    total = positive.sum(axis=1, min_count=1)
    result.loc[:, MAJOR_SHORT] = positive.div(total, axis=0) * 100.0
    return result


def application_filter(
    features: pd.DataFrame,
    *,
    sio2_max: float = SIO2_MAX,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """应用统一的太古代扩展池筛选条件。"""
    numeric = features[SHORT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missing_count = (numeric.isna() | numeric.le(0)).sum(axis=1)
    required = numeric[REQUIRED_MAJOR_SHORT].notna().all(axis=1)
    required &= numeric[REQUIRED_MAJOR_SHORT].gt(0).all(axis=1)
    normalized = normalize_major_elements(numeric)
    keep = (
        missing_count.lt(MAX_MISSING_EXCLUSIVE)
        & required
        & normalized["SIO2"].between(SIO2_MIN, sio2_max)
        & normalized["MGO"].le(MGO_MAX)
    )
    return keep, normalized, missing_count


def extract_georoc_craton(location: object) -> str:
    """从GeoROC层级LOCATION中提取最具体的克拉通、省或地盾名称。"""
    if pd.isna(location) or not str(location).strip():
        return "GeoROC_unmatched"

    parts = [part.strip() for part in str(location).split("/") if part.strip()]
    craton_parts = [part for part in parts if "CRATON" in part.upper()]
    if craton_parts:
        selected = craton_parts[-1]
    else:
        province_or_shield = [
            part
            for part in parts
            if "PROVINCE" in part.upper() or "SHIELD" in part.upper()
        ]
        selected = province_or_shield[0] if province_or_shield else parts[0]

    selected = selected.replace("_ARCHEAN", "")
    selected = selected.replace(" - ARCHEAN", "")
    return " ".join(selected.split()).strip()


def build_liu_pool(*, sio2_max: float = SIO2_MAX) -> pd.DataFrame:
    """构建SiO2上限放宽到54 wt%的Liu扩展池。"""
    data = pd.read_csv(LIU_RAW_PATH, low_memory=False)
    features = data[SHORT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    keep, normalized, missing_count = application_filter(
        features,
        sio2_max=sio2_max,
    )
    result = data.loc[keep].copy()
    result.loc[:, MAJOR_SHORT] = normalized.loc[keep, MAJOR_SHORT]
    result["missing_feature_count_36"] = missing_count.loc[keep].to_numpy()
    result["SOURCE_DATASET"] = "Liu_2024"
    result["SOURCE_ORIGINAL_TECTONIC_LABEL"] = np.nan
    result["SOURCE_AGE_TEXT"] = np.nan
    result["POOL_COMPONENT"] = np.where(
        result["SIO2"].le(53.0),
        "Liu_current_SiO2_le_53",
        "Liu_added_SiO2_53_to_54",
    )
    return result.reset_index(drop=True)


def build_georoc_recovered_pool(
    *,
    sio2_max: float = SIO2_MAX,
) -> tuple[pd.DataFrame, dict]:
    """恢复GeoROC中AGE文本含ARCHEAN且满足扩展应用QC的记录。"""
    ensure_import_paths()
    from iron_normalization import calculate_feot

    data = pd.read_csv(GEOROC_RAW_PATH, low_memory=False)
    data, _ = calculate_feot(data, allow_elemental_fe=False)
    age_text = data["AGE"].fillna("").astype(str)
    archean_mask = age_text.str.upper().str.contains("ARCHEAN", na=False)

    features = pd.DataFrame(index=data.index)
    for short_name, long_name in zip(SHORT_COLUMNS, LONG_COLUMNS):
        features[short_name] = pd.to_numeric(data[long_name], errors="coerce")
    keep_application, normalized, missing_count = application_filter(
        features,
        sio2_max=sio2_max,
    )
    keep = archean_mask & keep_application

    result = pd.DataFrame(index=data.index[keep])
    result["Craton"] = data.loc[keep, "LOCATION"].map(extract_georoc_craton).to_numpy()
    result["REFERENCE"] = data.loc[keep, "CITATIONS"].to_numpy()
    result["SAMPLE_ID"] = data.loc[keep, "SAMPLE NAME"].to_numpy()
    result["ROCK_NAME"] = data.loc[keep, "ROCK NAME"].to_numpy()
    result["LATITUDE"] = (
        pd.to_numeric(data.loc[keep, "LATITUDE MIN"], errors="coerce")
        + pd.to_numeric(data.loc[keep, "LATITUDE MAX"], errors="coerce")
    ).to_numpy() / 2.0
    result["LONGITUDE"] = (
        pd.to_numeric(data.loc[keep, "LONGITUDE MIN"], errors="coerce")
        + pd.to_numeric(data.loc[keep, "LONGITUDE MAX"], errors="coerce")
    ).to_numpy() / 2.0
    for column in ["MIN_AGE", "AGE", "MAX_AGE", "C_MIN_AGE", "C_AGE", "C_MAX_AGE", "Arc_probability3"]:
        result[column] = np.nan
    for short_name in SHORT_COLUMNS:
        result[short_name] = normalized.loc[keep, short_name].to_numpy()
    for optional in ["SC", "CU", "ZN", "GA", "CS", "TM", "PB", "U"]:
        long_name = f"{optional}(PPM)"
        result[optional] = (
            pd.to_numeric(data.loc[keep, long_name], errors="coerce").to_numpy()
            if long_name in data.columns else np.nan
        )
    result["missing_feature_count_36"] = missing_count.loc[keep].to_numpy()
    result["SOURCE_DATASET"] = "GeoROC_recovered"
    result["SOURCE_ORIGINAL_TECTONIC_LABEL"] = data.loc[keep, "TECTONIC SETTING"].to_numpy()
    result["SOURCE_AGE_TEXT"] = age_text.loc[keep].to_numpy()
    result["SOURCE_LOCATION"] = data.loc[keep, "LOCATION"].to_numpy()
    result["POOL_COMPONENT"] = "GeoROC_recovered_ARCHEAN"

    stats = {
        "raw_rows": len(data),
        "age_text_archean_rows": int(archean_mask.sum()),
        "application_qc_rows": int(keep.sum()),
    }
    return result.reset_index(drop=True), stats


def petdb_archean_stats() -> dict:
    """统计PetDB拆行合并前后的ARCHEAN年龄文本命中数。"""
    ensure_import_paths()
    from petdb_filter_analyzer import consolidate_petdb2_rows, DEFAULT_COLUMNS_TO_EXTRACT

    data = pd.read_csv(PETDB_RAW_PATH, low_memory=False)
    raw_age = (
        data["Geologic Age Prefix"].fillna("").astype(str)
        + " "
        + data["Geologic Age"].fillna("").astype(str)
    )
    raw_hits = int(raw_age.str.upper().str.contains("ARCHEAN", na=False).sum())
    consolidated, _ = consolidate_petdb2_rows(data, DEFAULT_COLUMNS_TO_EXTRACT)
    consolidated_age = (
        consolidated["Geologic Age Prefix"].fillna("").astype(str)
        + " "
        + consolidated["Geologic Age"].fillna("").astype(str)
    )
    consolidated_hits = int(
        consolidated_age.str.upper().str.contains("ARCHEAN", na=False).sum()
    )
    return {
        "raw_rows": len(data),
        "raw_age_text_archean_rows": raw_hits,
        "consolidated_rows": len(consolidated),
        "consolidated_age_text_archean_rows": consolidated_hits,
        "application_qc_rows": 0,
    }


def chemical_fingerprint(data: pd.DataFrame) -> pd.Series:
    """构造保守的化学组成指纹，用于删除完全重复记录。"""
    values = data[SHORT_COLUMNS].apply(pd.to_numeric, errors="coerce").round(3)
    return values.fillna(-999999.0).astype(str).agg("|".join, axis=1)


def combine_and_deduplicate(liu: pd.DataFrame, georoc: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """合并两个来源并按36元素完全一致的三位小数组成去重。"""
    combined = pd.concat([liu, georoc], ignore_index=True, sort=False)
    combined["_chemical_fingerprint"] = chemical_fingerprint(combined)
    duplicate_count = int(combined.duplicated("_chemical_fingerprint", keep="first").sum())
    combined = combined.drop_duplicates("_chemical_fingerprint", keep="first").copy()
    combined.drop(columns=["_chemical_fingerprint"], inplace=True)
    return combined.reset_index(drop=True), duplicate_count


def build_expanded_candidate_pool(
    *,
    sio2_max: float = SIO2_MAX,
    output_path: str | None = EXPANDED_RAW_PATH,
) -> tuple[pd.DataFrame, dict]:
    """按主项目逻辑构建Liu与GeoROC合并候选池，不执行模型推理。"""
    liu = build_liu_pool(sio2_max=sio2_max)
    georoc, georoc_stats = build_georoc_recovered_pool(sio2_max=sio2_max)
    expanded, duplicate_count = combine_and_deduplicate(liu, georoc)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        expanded.to_csv(output_path, index=False, encoding="utf-8-sig")

    stats = {
        "liu_rows": len(liu),
        "georoc_rows": len(georoc),
        "duplicate_rows": duplicate_count,
        "expanded_rows": len(expanded),
        **georoc_stats,
    }
    return expanded, stats


# 中文注释：以下旧实验辅助函数仅保留供历史追溯，正式入口不会调用它们。
def fit_and_transform_imputer(expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复用当前全局随机森林插补器，并返回无水标准化后的短列特征。"""
    imputation_module = _load_module_from_file(
        "imputation_train_predict",
        PROJECT_ROOT / "02_imputation" / "imputation_train_predict.py",
    )

    global_model = imputation_module.fit_global_model()
    long_features = pd.DataFrame(index=expanded.index)
    for short_name, long_name in zip(SHORT_COLUMNS, LONG_COLUMNS):
        long_features[long_name] = pd.to_numeric(expanded[short_name], errors="coerce")
    long_features = long_features.where(long_features > 0)
    imputed_long = imputation_module.transform_chemical_data(
        long_features, global_model, verbose=True
    )
    imputed_short = imputed_long.copy()
    imputed_short.columns = SHORT_COLUMNS
    normalized_short = normalize_major_elements(imputed_short)
    return imputed_long, normalized_short


def quantile_encode(normalized_short: pd.DataFrame) -> pd.DataFrame:
    """使用现代训练集固定分位数边界编码到1-255。"""
    with open(QUANTILE_PARAMS_PATH, "r", encoding="utf-8") as file:
        params = json.load(file)
    encoded = pd.DataFrame(index=normalized_short.index)
    for short_name, long_name in zip(SHORT_COLUMNS, LONG_COLUMNS):
        boundaries = np.asarray(params[long_name], dtype=float)
        encoded[long_name] = (
            np.searchsorted(
                boundaries,
                normalized_short[short_name].to_numpy(dtype=float),
                side="left",
            )
            + 1
        ).clip(1, 255).astype(np.int16)
    return encoded


def run_prediction(encoded: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """加载当前GeoDAN权重并预测九分类概率。"""
    ensure_import_paths()
    from domain_shift_diagnostics import (
        load_model_and_classes,
        predict_probabilities,
    )

    train_data = pd.read_csv(MODEL_TRAIN_PATH, low_memory=False)
    model, class_names, device, image_columns, sequence_columns = load_model_and_classes(
        train_data
    )
    probabilities = predict_probabilities(
        encoded,
        model,
        device,
        image_columns,
        sequence_columns,
    )
    result = pd.DataFrame(index=encoded.index)
    for index, class_name in enumerate(class_names):
        result[f"P_{class_name}"] = probabilities[:, index]
    predicted_index = probabilities.argmax(axis=1)
    result["Predicted_class"] = [class_names[index] for index in predicted_index]
    result["Prediction_confidence"] = probabilities.max(axis=1)
    arc_index = [class_names.index(class_name) for class_name in class_names if class_name in ARC_CLASSES]
    result["P_arc"] = probabilities[:, arc_index].sum(axis=1)
    result["High_arc_P_ge_0_5"] = result["P_arc"].ge(0.5)
    return result, probabilities, class_names


def add_domain_diagnostics(normalized_short: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    """计算扩展池相对现代训练集的kNN适用域比例。"""
    ensure_import_paths()
    from domain_shift_diagnostics import compute_distance_diagnostics

    modern = pd.read_csv(MODERN_TRAIN_CONTINUOUS_PATH, low_memory=False)
    modern_features = modern[LONG_COLUMNS].apply(pd.to_numeric, errors="coerce")
    modern_features.columns = SHORT_COLUMNS
    diagnostics, thresholds, _, _ = compute_distance_diagnostics(
        modern_features,
        normalized_short,
        modern["TECTONIC SETTING"].astype(str).to_numpy(),
    )
    prediction["knn_distance"] = diagnostics["knn_distance"].to_numpy()
    prediction["knn_outside_99"] = diagnostics["knn_outside_99"].to_numpy()
    prediction["High_arc_and_in_99_domain"] = (
        prediction["High_arc_P_ge_0_5"] & ~prediction["knn_outside_99"]
    )
    return thresholds


def main() -> None:
    ensure_import_paths()

    ARCHEAN_POOL_DIR.mkdir(parents=True, exist_ok=True)

    # 中文注释：正式流程到此只生成候选池，插补、分位数编码和预测由独立正式脚本负责。
    expanded, stats = build_expanded_candidate_pool()
    print(f"候选池已生成: {len(expanded)} 条")
    print(f"组成统计: {stats}")


if __name__ == "__main__":
    main()
