"""
==========================================================================
玄武岩地球化学缺失值全局插补：训练集拟合 + 测试集转换
==========================================================================

核心原则：
  1. 直接读取完整训练集，只拟合一套全局 scaler 和 MissForest。
  2. TECTONIC SETTING 是下游模型的预测目标，不参与插补模型拟合。
  3. 测试集使用同一个内存模型，只 transform，不重新 fit。
  4. TECTONIC SETTING 仅作为原始信息保留到输出文件中。
  5. 太古代应用集不插补，只保存原始缺失mask供GeoDAN显式编码。
==========================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_RAW_CSV, TEST_RAW_CSV,
    TRAIN_IMPUTED_CSV, TEST_IMPUTED_CSV,
    MASK_TRAIN_CSV, MASK_TEST_CSV,
    ARCHEAN_POOL_CSV, ARCHEAN_POOL_MASK_CSV,
)


# ============================================================================
# 默认配置（路径集中在 config/paths.py）
# ============================================================================

TRAIN_INPUT_CSV = str(TRAIN_RAW_CSV)
TRAIN_OUTPUT_CSV = str(TRAIN_IMPUTED_CSV)

TEST_INPUT_CSV = str(TEST_RAW_CSV)
TEST_OUTPUT_CSV = str(TEST_IMPUTED_CSV)

ARCHEAN_INPUT_CSV = str(ARCHEAN_POOL_CSV)

TRAIN_MISSING_MASK_CSV = str(MASK_TRAIN_CSV)
TEST_MISSING_MASK_CSV = str(MASK_TEST_CSV)
ARCHEAN_MISSING_MASK_CSV = str(ARCHEAN_POOL_MASK_CSV)

# 中文注释：数值插补流程保持一致，缺失信息只交给GeoDAN。
# 因此插补器自身默认不使用缺失指示特征。
USE_MISSING_INDICATORS_IN_IMPUTER = False

RF_PARAMS = {
    'n_estimators': 300,
    # 中文注释：内部遮挡验证中，使用80%的候选特征明显优于sqrt。
    # 不限制树深度，保留地球化学元素之间的非线性关系。
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.8,
    'bootstrap': True,
    # 中文注释：默认单进程，避免Windows受限环境创建并行进程时触发WinError 5。
    # 如需提速，可设置 BASALT_RF_N_JOBS 环境变量，例如 4。
    'n_jobs': max(1, int(os.environ.get('BASALT_RF_N_JOBS', '1'))),
    'random_state': 42,
}

CHEMICAL_COLUMNS = [
    'NA2O(WT%)', 'MGO(WT%)', 'AL2O3(WT%)', 'SIO2(WT%)', 'P2O5(WT%)',
    'K2O(WT%)', 'CAO(WT%)', 'TIO2(WT%)', 'MNO(WT%)', 'FEOT(WT%)',
    'RB(PPM)', 'V(PPM)', 'CR(PPM)', 'CO(PPM)', 'NI(PPM)', 'BA(PPM)',
    'SR(PPM)', 'Y(PPM)', 'ZR(PPM)', 'NB(PPM)', 'LA(PPM)', 'CE(PPM)',
    'PR(PPM)', 'ND(PPM)', 'SM(PPM)', 'EU(PPM)', 'GD(PPM)', 'TB(PPM)',
    'DY(PPM)', 'HO(PPM)', 'ER(PPM)', 'YB(PPM)', 'LU(PPM)', 'HF(PPM)',
    'TA(PPM)', 'TH(PPM)',
]

# 太古代文件使用短列名，插补时映射为与训练集一致的长列名。
ARCHEAN_COLUMN_MAPPING = {
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

# ============================================================================
# 数据预处理
# ============================================================================

def preprocess_chemical_data(
    df: pd.DataFrame,
    columns: List[str],
    missing_row_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    提取36个化学特征并转为数值。

    缺失比例超过80%的样品不进入插补流程；不执行 IQR 异常值清洗。
    """
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f'数据缺少化学特征列: {missing_columns}')

    chemical_data = df[columns].apply(pd.to_numeric, errors='coerce')
    min_non_null = int(len(columns) * (1 - missing_row_threshold))
    return chemical_data.dropna(thresh=min_non_null)


def clean_for_rf(X: pd.DataFrame) -> pd.DataFrame:
    """为随机森林准备特征输入：无穷值转为有限值，缺失值临时置0。"""
    X_clean = X.replace([np.inf, -np.inf], np.finfo(np.float32).max)
    return X_clean.fillna(0)


def build_rf_predictors(
    X_values: pd.DataFrame,
    X_missing: pd.DataFrame,
    feature_columns: List[str],
    use_missing_indicators: bool,
) -> pd.DataFrame:
    """组合数值和缺失指示编码，避免标准化后的真实0与临时填充值混淆。"""
    value_part = X_values[feature_columns].copy()
    if not use_missing_indicators:
        return value_part

    missing_part = X_missing[feature_columns].astype(np.uint8).copy()
    missing_part.columns = [
        f'missing_indicator__{column}' for column in feature_columns
    ]
    return pd.concat([value_part, missing_part], axis=1)


# ============================================================================
# 全局 MissForest
# ============================================================================

def missforest_fit(
    X_train: pd.DataFrame,
    rf_params: Optional[Dict] = None,
    use_missing_indicators: bool = USE_MISSING_INDICATORS_IN_IMPUTER,
) -> Tuple[Dict[str, RandomForestRegressor], List[str]]:
    """基于全部训练样品，为每个化学特征训练一个全局随机森林。"""
    if rf_params is None:
        rf_params = RF_PARAMS

    X_clean = clean_for_rf(X_train)
    X_missing = X_train.isna()
    column_order = list(X_clean.columns)
    imputers: Dict[str, RandomForestRegressor] = {}

    for index, column in enumerate(column_order, start=1):
        feature_columns = [name for name in column_order if name != column]
        observed_mask = X_train[column].notna()
        observed_count = int(observed_mask.sum())

        if observed_count == 0:
            print(f"  [跳过] 列 '{column}' 全空，无法训练")
            continue

        print(
            f'  [{index:02d}/{len(column_order)}] 训练 {column}，'
            f'有效目标样品 {observed_count}'
        )
        random_forest = RandomForestRegressor(**rf_params)
        predictors = build_rf_predictors(
            X_clean,
            X_missing,
            feature_columns,
            use_missing_indicators,
        )
        random_forest.fit(
            predictors.loc[observed_mask],
            X_train.loc[observed_mask, column],
        )
        imputers[column] = random_forest

    return imputers, column_order


def missforest_transform(
    X: pd.DataFrame,
    imputers: Dict[str, RandomForestRegressor],
    column_order: List[str],
    use_missing_indicators: bool = USE_MISSING_INDICATORS_IN_IMPUTER,
    verbose: bool = False,
) -> pd.DataFrame:
    """使用训练好的唯一一套全局模型填补缺失值，不重新拟合。"""
    X_ordered = X[column_order].copy()
    X_clean = clean_for_rf(X_ordered)
    X_imputed = X_clean.copy()
    X_missing = X_ordered.isna()
    imputed_count = 0

    for column in column_order:
        missing_mask = X_ordered[column].isna()
        if not missing_mask.any() or column not in imputers:
            continue

        feature_columns = [name for name in column_order if name != column]
        predictors = build_rf_predictors(
            X_imputed,
            X_missing,
            feature_columns,
            use_missing_indicators,
        )
        X_imputed.loc[missing_mask, column] = imputers[column].predict(
            predictors.loc[missing_mask]
        )
        imputed_count += int(missing_mask.sum())

    if verbose:
        print(f'  [插补] 共填补 {imputed_count} 个缺失值')

    return X_imputed


def fit_global_model() -> Dict:
    """
    从完整训练集提取化学特征，拟合唯一的全局 scaler 和 MissForest。

    这里不会读取 TECTONIC SETTING，因此目标标签无法参与插补模型训练。
    """
    data = pd.read_csv(TRAIN_INPUT_CSV)
    chemical_data = preprocess_chemical_data(data, CHEMICAL_COLUMNS)
    print(f'[训练输入] {TRAIN_INPUT_CSV}')
    print(f'[全局训练集] 原始 {len(data)} 条，有效样品 {len(chemical_data)} 条')

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(chemical_data),
        columns=CHEMICAL_COLUMNS,
        index=chemical_data.index,
    )

    print('[全局 MissForest] 训练36个随机森林模型')
    imputers, column_order = missforest_fit(
        X_scaled,
        use_missing_indicators=USE_MISSING_INDICATORS_IN_IMPUTER,
    )
    return {
        'imputers': imputers,
        'column_order': column_order,
        'scaler': scaler,
        'use_missing_indicators': USE_MISSING_INDICATORS_IN_IMPUTER,
    }


def transform_chemical_data(
    chemical_data: pd.DataFrame,
    global_model: Dict,
    verbose: bool = False,
) -> pd.DataFrame:
    """使用全局 scaler 和全局 MissForest 转换一批化学数据。"""
    scaler: StandardScaler = global_model['scaler']
    column_order = global_model['column_order']
    imputers = global_model['imputers']
    use_missing_indicators = global_model.get(
        'use_missing_indicators',
        False,
    )

    X_scaled = pd.DataFrame(
        scaler.transform(chemical_data[column_order]),
        columns=column_order,
        index=chemical_data.index,
    )
    X_imputed = missforest_transform(
        X_scaled,
        imputers,
        column_order,
        use_missing_indicators=use_missing_indicators,
        verbose=verbose,
    )
    restored = pd.DataFrame(
        scaler.inverse_transform(X_imputed),
        columns=column_order,
        index=chemical_data.index,
    )
    # 地球化学含量不应为负，避免随机森林外推产生少量负值。
    return restored.clip(lower=0.0)


def save_missing_mask(
    chemical_data: pd.DataFrame,
    output_path: str,
    dataset_name: str,
) -> pd.DataFrame:
    """保存36维原始缺失掩码，1表示原始缺失，0表示原始实测。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mask = chemical_data[CHEMICAL_COLUMNS].isna().astype(np.uint8)
    mask.columns = [
        f'missing_mask__{column}' for column in CHEMICAL_COLUMNS
    ]
    mask.reset_index(drop=True, inplace=True)
    mask.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(
        f'[{dataset_name}缺失mask] {output_path}，'
        f'shape={mask.shape}，缺失单元={int(mask.to_numpy().sum())}'
    )
    return mask


# ============================================================================
# 训练集和测试集输出
# ============================================================================

def build_output(
    source_data: pd.DataFrame,
    chemical_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
) -> pd.DataFrame:
    """用插补结果替换化学列，其余列（包括目标标签）保持原值和原顺序。"""
    result = source_data.loc[chemical_data.index].copy()
    result.loc[:, CHEMICAL_COLUMNS] = imputed_data[CHEMICAL_COLUMNS]
    return result.reset_index(drop=True)


def impute_trainset(global_model: Dict) -> pd.DataFrame:
    """使用全局模型插补完整训练集并输出。"""
    data = pd.read_csv(TRAIN_INPUT_CSV)
    chemical_data = preprocess_chemical_data(data, CHEMICAL_COLUMNS)
    save_missing_mask(
        chemical_data,
        TRAIN_MISSING_MASK_CSV,
        '训练集',
    )
    imputed_data = transform_chemical_data(
        chemical_data,
        global_model,
        verbose=True,
    )
    result = build_output(data, chemical_data, imputed_data)
    Path(TRAIN_OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(TRAIN_OUTPUT_CSV, index=False)
    print(f'[训练集输出] {TRAIN_OUTPUT_CSV}')
    print(f'[训练集输出] {len(result)} 条 × {result.shape[1]} 列')
    return result


def impute_testset(global_model: Dict) -> pd.DataFrame:
    """
    对测试集整体插补。

    不按 TECTONIC SETTING 分组，也不根据该标签选择任何模型。
    """
    print('=' * 70)
    print('[测试集] 使用唯一的全局 scaler 和 MissForest 整体插补')
    print('[防泄漏] TECTONIC SETTING 不参与 fit、transform 或模型选择')
    print('=' * 70)

    data = pd.read_csv(TEST_INPUT_CSV)
    chemical_data = preprocess_chemical_data(data, CHEMICAL_COLUMNS)
    save_missing_mask(
        chemical_data,
        TEST_MISSING_MASK_CSV,
        '测试集',
    )
    print(f'[测试输入] 原始 {len(data)} 条，有效插补样品 {len(chemical_data)} 条')

    imputed_data = transform_chemical_data(
        chemical_data,
        global_model,
        verbose=True,
    )
    result = build_output(data, chemical_data, imputed_data)

    Path(TEST_OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(TEST_OUTPUT_CSV, index=False)
    print(f'[测试集输出] {TEST_OUTPUT_CSV}')
    print(f'[测试集输出] {len(result)} 条 × {result.shape[1]} 列')

    remaining_nan = int(result[CHEMICAL_COLUMNS].isna().sum().sum())
    if remaining_nan:
        print(f'[警告] 化学特征仍有 {remaining_nan} 个缺失值')
    else:
        print('[验证] 化学特征无残留缺失值')

    return result


def save_archean_mask_only() -> Optional[pd.DataFrame]:
    """太古代数据不插补，只保存原始36维缺失mask。"""
    print('=' * 70)
    print('[太古代应用集] 不执行插补，仅生成36维原始缺失mask')
    print('=' * 70)

    if not Path(ARCHEAN_INPUT_CSV).exists():
        print(
            f'[跳过] 未找到扩展太古代应用集: {ARCHEAN_INPUT_CSV}\n'
            '        请先运行 06_archean_application/extended_archean_pool_analysis.py '
            '构建 3,483 条应用集后重试；正式预测脚本也会自行生成该 mask。'
        )
        return None

    data = pd.read_csv(ARCHEAN_INPUT_CSV)
    if len(data) != 3483:
        raise ValueError(f'扩展太古代输入应为3483条，实际为 {len(data)} 条')

    missing_columns = [
        column for column in ARCHEAN_COLUMN_MAPPING if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f'太古代数据缺少化学特征列: {missing_columns}')

    chemical_data = pd.DataFrame(index=data.index)
    for short_name, long_name in ARCHEAN_COLUMN_MAPPING.items():
        chemical_data[long_name] = pd.to_numeric(
            data[short_name],
            errors='coerce',
        )

    # 中文注释：太古代筛选流程将0和负值定义为缺失，预测时数值通道编码为0。
    chemical_data = chemical_data[CHEMICAL_COLUMNS].where(
        chemical_data[CHEMICAL_COLUMNS] > 0
    )
    mask = save_missing_mask(
        chemical_data,
        ARCHEAN_MISSING_MASK_CSV,
        '太古代应用集',
    )
    print(f'[太古代验证] 保留原始样品 {len(data)} 条，不生成插补版CSV')
    return mask


def cached_modern_outputs_are_valid() -> bool:
    """中文注释：已有结果与当前切分集一致时，避免重复训练36个随机森林。"""
    if os.environ.get("BASALT_REUSE_IMPUTATION_CACHE", "1") == "0":
        return False

    output_paths = [
        Path(TRAIN_OUTPUT_CSV),
        Path(TEST_OUTPUT_CSV),
        Path(TRAIN_MISSING_MASK_CSV),
        Path(TEST_MISSING_MASK_CSV),
    ]
    if not all(path.exists() for path in output_paths):
        return False

    try:
        for input_path, output_path in [
            (TRAIN_INPUT_CSV, TRAIN_OUTPUT_CSV),
            (TEST_INPUT_CSV, TEST_OUTPUT_CSV),
        ]:
            source = pd.read_csv(input_path, low_memory=False)
            cached = pd.read_csv(output_path, low_memory=False)
            if len(source) != len(cached):
                return False

            if not set(CHEMICAL_COLUMNS).issubset(cached.columns):
                return False
            metadata_columns = [
                column for column in source.columns
                if column not in CHEMICAL_COLUMNS
            ]
            if not set(metadata_columns).issubset(cached.columns):
                return False
            source_metadata = source[metadata_columns].fillna("<NA>").astype(str)
            cached_metadata = cached[metadata_columns].fillna("<NA>").astype(str)
            if not source_metadata.equals(cached_metadata):
                return False
        return True
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


if __name__ == '__main__':
    print('=' * 70)
    print('玄武岩地球化学缺失值全局插补：训练集 fit -> 测试集 transform')
    print('TECTONIC SETTING 仅保留到输出，不参与插补')
    print('太古代应用集固定不插补，只生成缺失mask')
    print('=' * 70)

    if cached_modern_outputs_are_valid():
        print('[缓存] 训练集/测试集插补结果与当前切分集一致，跳过重复拟合。')
    else:
        model = fit_global_model()
        impute_trainset(model)
        impute_testset(model)
    save_archean_mask_only()

    print('[完成] 现代训练/测试集插补与太古代缺失mask生成完成')
