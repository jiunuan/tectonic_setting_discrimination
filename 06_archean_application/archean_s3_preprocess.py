"""太古代玄武岩应用口径筛选 + 六案例区统一预处理。

本脚本负责两部分：

一、统一应用口径筛选（`preprocess_archean`，被多个脚本复用）：
1. 提取36个地球化学特征；
2. 按 NaN、0 或负值统计缺失数，保留缺失数小于18的样品；
3. 要求 SiO2、Al2O3、FeOT、MgO、CaO 为有效正值；
4. 对10个主量元素进行逐行无水标准化（仅用正值主量参与求和）；
5. 保留标准化后 SiO2 为44-53 wt%、MgO不超过18 wt%的样品。

`preprocess_archean()` 同时被正式预测脚本
archean_vit_transformer_dualstream_predict_analysis.py 复用，
用于 Liu 全库与 6 克拉通案例数据的统一预处理。

二、六个案例区配置与预处理（供案例研究图与预测流程复用）：
- `CASE_STUDIES_ORDER` / `CASE_STUDY_TITLES`：六个案例区的标签、展示名与代表年龄；
- `CaseStudyConfig` / `build_case_study_configs()`：每个案例的输入与输出路径配置；
- `preprocess_case_study()`：读取、区域定位（克拉通/年龄/文献关键词）、保存原始表，
  并按统一44-53 wt%口径筛选单个案例；
- `load_final_age_constrained_pool()`：读取正式年龄约束总池并执行同口径筛选。

注意：候选池（3,483 条，SiO2≤54 wt%）的构建入口是
extended_archean_pool_analysis.py；正式太古代应用集与现代训练集口径一致，
统一为无水 SiO2≤53 wt%，由 `load_final_age_constrained_pool()` 对候选池
筛选得到 3,012 条（EXPECTED_FINAL_COUNT）。本脚本 main() 复现 Liu 数据严格
口径（SiO2≤53）的 2,116 条筛选结果，并对六个案例区做统一预处理。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    ARCHEAN_S3_CSV,
    ARCHEAN_DATA_SUBDIR,
    ARCHEAN_CASE_DIR,
    ARCHEAN_POOL_CSV,
    ZENODO_ARCHEAN_CSV,
)


# ============================================================================
# 配置（路径集中在 config/paths.py）
# ============================================================================

SOURCE_CSV = str(ARCHEAN_S3_CSV)

OUTPUT_CSV = str(ARCHEAN_DATA_SUBDIR / 'archean_basalt_filtered_2116.csv')

MAX_MISSING_FEATURES_EXCLUSIVE = 18
SIO2_MIN = 44.0
SIO2_MAX = 53.0
MGO_MAX = 18.0
EXPECTED_SAMPLE_COUNT = 2116
# 中文注释：正式太古代应用集与现代训练集一致，固定为无水 SiO2≤53 wt% 口径；
# 候选池（extended_archean_pool_analysis.py，SiO2≤54）经本口径筛选后为 3012 条。
EXPECTED_FINAL_COUNT = 3012

SHORT_CHEMICAL_COLUMNS = [
    'NA2O', 'MGO', 'AL2O3', 'SIO2', 'P2O5', 'K2O', 'CAO', 'TIO2',
    'MNO', 'FEOT', 'RB', 'V', 'CR', 'CO', 'NI', 'BA', 'SR', 'Y',
    'ZR', 'NB', 'LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB',
    'DY', 'HO', 'ER', 'YB', 'LU', 'HF', 'TA', 'TH',
]

MAJOR_COLUMNS = [
    'SIO2', 'TIO2', 'AL2O3', 'FEOT', 'MNO',
    'MGO', 'CAO', 'NA2O', 'K2O', 'P2O5',
]

REQUIRED_MAJOR_COLUMNS = ['SIO2', 'AL2O3', 'FEOT', 'MGO', 'CAO']

CHEMICAL_COLUMN_MAPPING = {
    'SIO2': 'SIO2(WT%)', 'TIO2': 'TIO2(WT%)',
    'AL2O3': 'AL2O3(WT%)', 'FEOT': 'FEOT(WT%)',
    'MNO': 'MNO(WT%)', 'MGO': 'MGO(WT%)', 'CAO': 'CAO(WT%)',
    'NA2O': 'NA2O(WT%)', 'K2O': 'K2O(WT%)', 'P2O5': 'P2O5(WT%)',
    'V': 'V(PPM)', 'CR': 'CR(PPM)', 'CO': 'CO(PPM)', 'NI': 'NI(PPM)',
    'RB': 'RB(PPM)', 'SR': 'SR(PPM)', 'Y': 'Y(PPM)', 'ZR': 'ZR(PPM)',
    'NB': 'NB(PPM)', 'BA': 'BA(PPM)', 'LA': 'LA(PPM)', 'CE': 'CE(PPM)',
    'PR': 'PR(PPM)', 'ND': 'ND(PPM)', 'SM': 'SM(PPM)', 'EU': 'EU(PPM)',
    'GD': 'GD(PPM)', 'TB': 'TB(PPM)', 'DY': 'DY(PPM)', 'HO': 'HO(PPM)',
    'ER': 'ER(PPM)', 'YB': 'YB(PPM)', 'LU': 'LU(PPM)', 'HF': 'HF(PPM)',
    'TA': 'TA(PPM)', 'TH': 'TH(PPM)',
}


def read_csv_fallback(file_path: Path | str) -> pd.DataFrame:
    """兼容 UTF-8 与 UTF-8 BOM 读取 CSV。"""
    try:
        return pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')


def extract_chemical_features(data: pd.DataFrame) -> pd.DataFrame:
    """兼容短列名和标准长列名，提取36个化学特征并转为数值。"""
    features = pd.DataFrame(index=data.index)
    missing_columns = []
    for short_name in SHORT_CHEMICAL_COLUMNS:
        long_name = CHEMICAL_COLUMN_MAPPING[short_name]
        if short_name in data.columns:
            source_column = short_name
        elif long_name in data.columns:
            source_column = long_name
        else:
            missing_columns.append(f'{short_name}/{long_name}')
            continue
        features[short_name] = pd.to_numeric(
            data[source_column],
            errors='coerce',
        )

    if missing_columns:
        raise ValueError(f'太古代原始数据缺少化学特征列: {missing_columns}')

    return features[SHORT_CHEMICAL_COLUMNS]


def normalize_major_elements(features: pd.DataFrame) -> pd.DataFrame:
    """仅使用正值主量元素进行逐行无水标准化到100 wt%。"""
    normalized = features.copy()
    positive_majors = normalized[MAJOR_COLUMNS].where(
        normalized[MAJOR_COLUMNS] > 0
    )
    major_totals = positive_majors.sum(axis=1, min_count=1)
    invalid_mask = major_totals.isna() | (major_totals <= 0)
    if invalid_mask.any():
        raise ValueError(
            f'存在 {int(invalid_mask.sum())} 条样品没有有效主量元素总和'
        )

    normalized.loc[:, MAJOR_COLUMNS] = positive_majors.div(
        major_totals,
        axis=0,
    ) * 100.0
    return normalized


def application_filter(
    features: pd.DataFrame,
    *,
    sio2_min: float = SIO2_MIN,
    sio2_max: float = SIO2_MAX,
    mgo_max: float = MGO_MAX,
    max_missing_exclusive: int = MAX_MISSING_FEATURES_EXCLUSIVE,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """执行与主项目一致的完整度、关键主量和玄武岩范围筛选。"""
    numeric = features[SHORT_CHEMICAL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    missing_count = (numeric.isna() | numeric.le(0)).sum(axis=1)
    required = numeric[REQUIRED_MAJOR_COLUMNS].notna().all(axis=1)
    required &= numeric[REQUIRED_MAJOR_COLUMNS].gt(0).all(axis=1)
    normalized = normalize_major_elements(numeric)
    keep = (
        missing_count.lt(max_missing_exclusive)
        & required
        & normalized["SIO2"].between(sio2_min, sio2_max)
        & normalized["MGO"].le(mgo_max)
    )
    return keep, normalized, missing_count


def preprocess_archean(
    data: pd.DataFrame,
    expected_sample_count: int | None = None,
    *,
    dataset_name: str = "太古代样品",
    sio2_min: float = SIO2_MIN,
    sio2_max: float = SIO2_MAX,
    mgo_max: float = MGO_MAX,
) -> pd.DataFrame:
    """按主项目正式口径筛选全库、Liu 或案例区数据。"""
    features = extract_chemical_features(data)
    missing_counts = (features.isna() | features.le(0)).sum(axis=1)
    keep_missing = missing_counts.lt(MAX_MISSING_FEATURES_EXCLUSIVE)

    required_values = features.loc[
        keep_missing,
        REQUIRED_MAJOR_COLUMNS,
    ]
    keep_required_local = required_values.notna().all(axis=1)
    keep_required_local &= required_values.gt(0).all(axis=1)
    required_indices = required_values.index[keep_required_local]

    # 中文注释：只对通过关键主量检查的样品执行无水标准化。
    normalized_features = normalize_major_elements(features.loc[required_indices])
    keep_basalt = (
        normalized_features["SIO2"].between(sio2_min, sio2_max)
        & normalized_features["MGO"].le(mgo_max)
    )
    final_features = normalized_features.loc[keep_basalt].copy()
    final_indices = final_features.index

    result = data.loc[final_indices].copy()
    for column in SHORT_CHEMICAL_COLUMNS:
        if column in MAJOR_COLUMNS:
            result[column] = final_features[column]
        elif column not in result.columns:
            result[column] = features.loc[final_indices, column]
    result.loc[:, MAJOR_COLUMNS] = final_features[MAJOR_COLUMNS]
    result["missing_feature_count_36"] = missing_counts.loc[
        final_indices
    ].to_numpy()
    result.reset_index(drop=True, inplace=True)

    print(f"[{dataset_name}] 原始样品: {len(data)}")
    print(
        f"[{dataset_name}] 缺失数 < {MAX_MISSING_FEATURES_EXCLUSIVE}: "
        f"{int(keep_missing.sum())}"
    )
    print(f"[{dataset_name}] 五项关键主量有效: {len(required_indices)}")
    print(
        f"[{dataset_name}] 无水SiO2={sio2_min:g}-{sio2_max:g}, "
        f"MgO<={mgo_max:g}: {len(result)}"
    )

    if expected_sample_count is not None and len(result) != expected_sample_count:
        raise ValueError(
            f'{dataset_name}筛选结果应为 {expected_sample_count} 条，'
            f'实际为 {len(result)} 条'
        )
    return result


def load_final_age_constrained_pool(
    *,
    expected_sample_count: int | None = EXPECTED_FINAL_COUNT,
) -> pd.DataFrame:
    """读取年龄约束候选池（ARCHEAN_POOL_CSV）并执行正式 SiO2≤53 wt% 筛选。

    候选池为 SiO2≤54 的 3,483 条；经与训练集一致的 44-53 wt% 口径筛选后，
    得到正式应用集 3,012 条（与 EXPECTED_FINAL_COUNT 一致）。
    """
    # 中文注释：正式候选池未复制到流程输出目录时，回退读取 Zenodo 精简发布的太古代应用表。
    pool_csv = ZENODO_ARCHEAN_CSV if Path(ZENODO_ARCHEAN_CSV).exists() else ARCHEAN_POOL_CSV
    data = read_csv_fallback(pool_csv)
    return preprocess_archean(
        data,
        expected_sample_count=expected_sample_count,
        dataset_name='正式太古代总池',
    )


# ============================================================================
# 六个案例区配置与筛选
# ============================================================================

# 中文注释：案例区按代表年龄从老到新排列，标签即预测/图件文件名前缀。
CASE_STUDIES_ORDER = [
    ('Isua', 'Isua', 3.75),
    ('Pilbara', 'Pilbara', 3.30),
    ('Ivisaartoq', 'Ivisaartoq', 3.05),
    ('Norseman_Kambalda', 'Norseman-Kambalda', 2.69),
    ('Abitibi', 'Abitibi', 2.70),
    ('North_China_Craton', 'North China Craton', 2.55),
]
CASE_STUDY_TITLES = {
    case_label: case_title
    for case_label, case_title, _ in CASE_STUDIES_ORDER
}

# 中文注释：案例区输入/输出目录全部相对 config/paths.py 推导。
CASE_SOURCE_DIR = Path(str(ARCHEAN_DATA_SUBDIR))
CASE_STUDY_OUTPUT_ROOT = Path(str(ARCHEAN_CASE_DIR))
CASE_RAW_OUTPUT_DIR = CASE_STUDY_OUTPUT_ROOT / 'raw'
CASE_PREPROCESSED_OUTPUT_DIR = CASE_STUDY_OUTPUT_ROOT / 'preprocessed'
CASE_PREDICTIONS_OUTPUT_DIR = CASE_STUDY_OUTPUT_ROOT / 'predictions'
CASE_SUMMARY_CSV_PATH = CASE_PREDICTIONS_OUTPUT_DIR / 'case_study_summary.csv'
# 中文注释：左栏案例组成热力图、右栏高弧KDE山脊图的组合主图。
CASE_FIG_COMBINED_PATH = (
    CASE_PREDICTIONS_OUTPUT_DIR / 'fig_case_studies_bars_ridgeline.png'
)


@dataclass(frozen=True)
class CaseStudyConfig:
    """单个案例区的输入与预处理输出配置。"""

    case_label: str
    case_title: str
    approx_age_ga: float
    source_path: Path
    raw_output_path: Path
    preprocessed_path: Path
    predictions_path: Path
    # 中文注释：以下字段仅用于"从正式总池抽取地质单元"的案例（如 Ivisaartoq）。
    source_craton: str | None = None
    age_min_ma: float | None = None
    age_max_ma: float | None = None
    reference_keyword: str | None = None


def build_case_study_configs() -> list[CaseStudyConfig]:
    """生成六个案例的显式配置，分别保存原始、预处理和预测结果。"""

    def _case(
        case_label: str,
        case_title: str,
        approx_age_ga: float,
        source_path: Path,
        *,
        source_craton: str | None = None,
        age_min_ma: float | None = None,
        age_max_ma: float | None = None,
        reference_keyword: str | None = None,
    ) -> CaseStudyConfig:
        return CaseStudyConfig(
            case_label,
            case_title,
            approx_age_ga,
            source_path,
            CASE_RAW_OUTPUT_DIR / f'{case_label}_raw.csv',
            CASE_PREPROCESSED_OUTPUT_DIR / f'{case_label}_preprocessed.csv',
            CASE_PREDICTIONS_OUTPUT_DIR / f'{case_label}_predictions.csv',
            source_craton=source_craton,
            age_min_ma=age_min_ma,
            age_max_ma=age_max_ma,
            reference_keyword=reference_keyword,
        )

    return [
        _case('Isua', 'Isua', 3.75, CASE_SOURCE_DIR / 'Isua.csv'),
        _case('Pilbara', 'Pilbara', 3.30, CASE_SOURCE_DIR / 'Pilbara.csv'),
        # 中文注释：Ivisaartoq 没有独立原始表，从正式总池按克拉通+年龄+文献关键词定位。
        _case(
            'Ivisaartoq', 'Ivisaartoq', 3.05,
            Path(str(ARCHEAN_POOL_CSV)),
            source_craton='North Atlantic Craton',
            age_min_ma=3000.0,
            age_max_ma=3100.0,
            reference_keyword='IVISAARTOQ',
        ),
        _case(
            'Norseman_Kambalda', 'Norseman-Kambalda', 2.69,
            CASE_SOURCE_DIR / 'Norseman&Kambalda.csv',
        ),
        _case('Abitibi', 'Abitibi', 2.70, CASE_SOURCE_DIR / 'Superior_Abitibi.csv'),
        _case(
            'North_China_Craton', 'North China Craton', 2.55,
            CASE_SOURCE_DIR / 'North_China_Craton.csv',
        ),
    ]


def preprocess_case_study(
    case_config: CaseStudyConfig,
) -> pd.DataFrame | None:
    """读取、区域定位、保存并按统一44-53 wt%规则筛选一个案例区。"""
    if not case_config.source_path.exists():
        print(f'  [警告] 缺少案例数据: {case_config.source_path}')
        return None

    raw_data = read_csv_fallback(case_config.source_path)

    # 中文注释：正式池案例先按克拉通、年龄和文献关键词定位具体地质单元。
    if case_config.source_craton is not None:
        if 'Craton' not in raw_data.columns:
            raise ValueError(f'{case_config.case_title}数据缺少Craton列')
        raw_data = raw_data.loc[
            raw_data['Craton'].eq(case_config.source_craton)
        ].copy()

    if case_config.age_min_ma is not None or case_config.age_max_ma is not None:
        if 'C_AGE' not in raw_data.columns or 'AGE' not in raw_data.columns:
            raise ValueError(f'{case_config.case_title}数据缺少C_AGE或AGE列')
        age_ma = pd.to_numeric(raw_data['C_AGE'], errors='coerce')
        age_ma = age_ma.fillna(pd.to_numeric(raw_data['AGE'], errors='coerce'))
        age_keep = pd.Series(True, index=raw_data.index)
        if case_config.age_min_ma is not None:
            age_keep &= age_ma.ge(case_config.age_min_ma)
        if case_config.age_max_ma is not None:
            age_keep &= age_ma.le(case_config.age_max_ma)
        raw_data = raw_data.loc[age_keep].copy()

    if case_config.reference_keyword is not None:
        if 'REFERENCE' not in raw_data.columns:
            raise ValueError(f'{case_config.case_title}数据缺少REFERENCE列')
        reference_keep = raw_data['REFERENCE'].fillna('').astype(str).str.contains(
            case_config.reference_keyword,
            case=False,
            regex=False,
        )
        raw_data = raw_data.loc[reference_keep].copy()

    if raw_data.empty:
        raise ValueError(f'{case_config.case_title}区域筛选后没有样品')

    case_config.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(
        case_config.raw_output_path,
        index=False,
        encoding='utf-8-sig',
    )

    # 中文注释：复用统一预处理流程，严格执行缺失数<18、五项主量有效、
    # 主量无水标准化及SiO2=44-53 wt%、MgO<=18 wt%的筛选。
    preprocessed = preprocess_archean(
        raw_data,
        dataset_name=case_config.case_title,
    )
    case_config.preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed.to_csv(
        case_config.preprocessed_path,
        index=False,
        encoding='utf-8-sig',
    )
    return preprocessed


def preprocess_all_case_studies() -> dict[str, pd.DataFrame]:
    """批量筛选六个案例区并返回结果。"""
    results: dict[str, pd.DataFrame] = {}
    for case_config in build_case_study_configs():
        result = preprocess_case_study(case_config)
        if result is not None:
            results[case_config.case_label] = result
    return results


def main() -> None:
    """复现 Liu 严格池 2116 条结果，并对六个案例区做统一预处理。"""
    data = read_csv_fallback(SOURCE_CSV)
    result = preprocess_archean(
        data,
        expected_sample_count=EXPECTED_SAMPLE_COUNT,
        dataset_name='Liu',
    )
    result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    major_sum = result[MAJOR_COLUMNS].sum(axis=1, min_count=1)
    print(f'[输出] {OUTPUT_CSV}')
    print(f'[输出形状] {result.shape[0]} 条 × {result.shape[1]} 列')
    print(
        f'[主量总和范围] {major_sum.min():.6f}-'
        f'{major_sum.max():.6f}'
    )

    print('=' * 72)
    print('六个案例区预处理：')
    case_results = preprocess_all_case_studies()
    for case_label, _, _ in CASE_STUDIES_ORDER:
        if case_label in case_results:
            print(f'  {case_label}: {len(case_results[case_label])} 条')


if __name__ == '__main__':
    main()
