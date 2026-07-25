import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import GEOROC_FILTERED_CSV, PETDB_FILTERED_CSV, COMBINED_CSV

# 待合并的两个筛选结果（GEOROC + PetDB），路径由 config/paths.py 提供
file_paths = [
    str(GEOROC_FILTERED_CSV),
    str(PETDB_FILTERED_CSV),
]

# 创建空列表存储读取的DataFrame
df_list = []

# 尝试不同的编码格式和处理读取错误
encodings = ['utf-8', 'latin1', 'ISO-8859-1']

for file in file_paths:
    file_read = False
    for encoding in encodings:
        try:
            with open(file, 'r', encoding=encoding) as f:
                df = pd.read_csv(f)
                df_list.append(df)
                file_read = True
                break
        except UnicodeDecodeError:
            print(f"Error reading {file} with encoding {encoding}. Trying next encoding.")
        except Exception as e:
            print(f"Unexpected error reading {file} with encoding {encoding}: {e}")
    if not file_read:
        print(f"Failed to read {file} with all tried encodings.")

# 合并所有 DataFrame
if not df_list:
    raise FileNotFoundError(
        "未找到可合并的筛选结果。请先运行 extract_georoc.py 和 extract_petdb.py，"
        "或检查 data/02_filtered 下的筛选文件名。"
    )
combined_df = pd.concat(df_list, ignore_index=True)

# 保存合并后的 DataFrame
COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
combined_df.to_csv(COMBINED_CSV, index=False)
print("合并成功")

