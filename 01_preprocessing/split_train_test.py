import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import COMBINED_CSV, ZENODO_MODERN_CSV, SPLIT_DIR


# =========================
# 默认配置
# 直接修改这里即可，不需要命令行参数
# =========================
# 中文注释：默认读取Zenodo精简发布表，保证从训练/测试切分开始即可复现。
# 若要把本次运行的GEOROC+PetDB合并结果接入下游，设置 BASALT_USE_COMBINED=1。
USE_REBUILT_COMBINED = os.environ.get("BASALT_USE_COMBINED", "0") == "1"
if USE_REBUILT_COMBINED and COMBINED_CSV.exists():
    DEFAULT_INPUT_PATH = COMBINED_CSV
else:
    DEFAULT_INPUT_PATH = (
        ZENODO_MODERN_CSV
        if ZENODO_MODERN_CSV.exists()
        else COMBINED_CSV
    )
INPUT_FILE = str(DEFAULT_INPUT_PATH)
OUTPUT_DIR = str(SPLIT_DIR)
LABEL_COLUMN = "TECTONIC SETTING"

# 测试集比例
TEST_SIZE = 0.2

# 验证集比例，默认不切验证集；如果需要可改成 0.1 之类
VAL_SIZE = 0.0

# 随机种子，保证每次切分结果可复现
RANDOM_STATE = 32

# 是否按标签分层切分
USE_STRATIFY = True


def validate_config():
    """检查输入路径和切分比例是否合理。"""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        raise ValueError("TEST_SIZE 必须在 0 和 1 之间。")

    if VAL_SIZE < 0 or VAL_SIZE >= 1:
        raise ValueError("VAL_SIZE 必须在 [0, 1) 范围内。")

    if TEST_SIZE + VAL_SIZE >= 1:
        raise ValueError("TEST_SIZE + VAL_SIZE 必须小于 1。")


def print_split_summary(name, df, label_column):
    """打印每个子集的样本数和类别分布。"""
    print(f"\n[{name}] 样本数: {len(df)}")
    if label_column in df.columns:
        counts = df[label_column].value_counts().sort_index()
        ratios = (counts / len(df) * 100).round(2)
        summary = pd.DataFrame({"count": counts, "percent": ratios})
        print(summary.to_string())


def save_summary(summary_rows, output_dir):
    """保存切分汇总表，方便后面核对类别比例。"""
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "split_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总文件已保存: {summary_path}")


def main():
    """按默认配置将总表切分为训练集/测试集（可选验证集）。"""
    validate_config()

    # 读取总表
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"未找到标签列: {LABEL_COLUMN}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 如果启用分层切分，则使用标签列保持类别比例稳定
    stratify_labels = df[LABEL_COLUMN] if USE_STRATIFY else None
    # 中文注释：无论输入来自 Zenodo 公开表还是旧的合并表，
    # 输出名称都固定为下游插补脚本约定的 01_basalt_number_year_*。
    base_name = "01_basalt_number_year"

    # 如果设置了验证集比例，则先切出 train 和 temp，再把 temp 切成 val/test
    if VAL_SIZE > 0:
        train_df, temp_df = train_test_split(
            df,
            test_size=TEST_SIZE + VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_labels,
        )

        temp_stratify = temp_df[LABEL_COLUMN] if USE_STRATIFY else None
        relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=RANDOM_STATE,
            stratify=temp_stratify,
        )

        split_map = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
    else:
        # 默认只切训练集和测试集
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_labels,
        )
        split_map = {
            "train": train_df,
            "test": test_df,
        }

    # 保存各个子集，并记录类别统计
    summary_rows = []
    for split_name, split_df in split_map.items():
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{split_name}.csv")
        split_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"{split_name} 数据已保存: {output_path}")
        print_split_summary(split_name, split_df, LABEL_COLUMN)

        label_counts = split_df[LABEL_COLUMN].value_counts().sort_index()
        for label, count in label_counts.items():
            summary_rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "percent": round(count / len(split_df) * 100, 4),
                }
            )

    save_summary(summary_rows, OUTPUT_DIR)
    print(f"\n切分完成，随机种子: {RANDOM_STATE}")


if __name__ == "__main__":
    main()
