"""使用人工核验的本地映射统一太古代数据中的克拉通名称。"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


# 中文注释：默认路径均为仓库内的相对路径，脚本不依赖网络或 API。
DEFAULT_INPUT = Path(
    "data/archean/outputs/archean_geodan_final/expanded_archean_predictions.csv"
)
DEFAULT_OUTPUT = Path(
    "data/archean/outputs/archean_geodan_final/expanded_archean_predictions_standardized.csv"
)

# 中文注释：这里只收录已人工核验的确定性归并关系，未列出的名称保持原值。
CRATON_NAME_MAP = {
    "abitibi": "Superior Craton",
    "abitibi greenstone belt": "Superior Craton",
    "wawa": "Superior Craton",
    "wawa subprovince": "Superior Craton",
    "uchi": "Superior Craton",
    "uchi subprovince": "Superior Craton",
    "pontiac": "Superior Craton",
    "pontiac subprovince": "Superior Craton",
    "barberton": "Kaapvaal Craton",
    "barberton greenstone belt": "Kaapvaal Craton",
    "isua": "North Atlantic Craton",
    "isua greenstone belt": "North Atlantic Craton",
    "southwest greenland": "North Atlantic Craton",
    "southwestern greenland": "North Atlantic Craton",
    "sw greenland": "North Atlantic Craton",
    "eastern dharwar": "Dharwar Craton",
    "eastern dharwar craton": "Dharwar Craton",
    "western dharwar": "Dharwar Craton",
    "western dharwar craton": "Dharwar Craton",
}


def normalize_lookup_key(value: str) -> str:
    """生成仅用于字典查询的稳定键。"""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def standardize_craton_name(value: Any) -> Any:
    """映射已核验别名；空值和未收录名称保持不变。"""
    if pd.isna(value):
        return value
    return CRATON_NAME_MAP.get(normalize_lookup_key(str(value)), value)


def standardize_craton_dataframe(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """标准化 Craton 列，并保留原始名称列。"""
    if "Craton" not in data.columns:
        raise ValueError("输入 CSV 缺少必需的 Craton 列。")
    output = data.copy()
    if "Craton_original" not in output.columns:
        position = output.columns.get_loc("Craton")
        output.insert(position, "Craton_original", output["Craton"].to_numpy(copy=True))
    standardized = output["Craton_original"].map(standardize_craton_name)
    changed = int(
        (
            standardized.fillna("<NA>").astype(str)
            != output["Craton_original"].fillna("<NA>").astype(str)
        ).sum()
    )
    output["Craton"] = standardized
    return output, changed


def main() -> None:
    """运行离线标准化。"""
    parser = argparse.ArgumentParser(
        description="Geographic names were standardized and manually verified."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    output, changed = standardize_craton_dataframe(data)
    print(f"输入样品数: {len(data)}")
    print(f"标准化替换数: {changed}")
    if args.check_only:
        print("检查模式：未写入文件。")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
