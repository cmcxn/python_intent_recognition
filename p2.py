import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedGroupKFold

# 1. 合并并做“规范化文本”作为 group 键
train_df = pd.read_csv("data/train.csv")
dev_df   = pd.read_csv("data/test.csv")
df = pd.concat([train_df, dev_df], ignore_index=True)

def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)        # 去空白
    return s

df["norm_text"] = df["text"].apply(normalize_text)

# 2. 同文本不同标签的冲突检查（如有需人工处理）
conflict = df.groupby("norm_text")["label"].nunique()
conflict_keys = conflict[conflict > 1].index.tolist()
print("Conflicting groups:", len(conflict_keys))  # 正常应为 0
# 若 >0，先人工检查这些条目后统一标签或删除异常样本

# 3. 去重（同文本同标签只保留一条）
df = df.drop_duplicates(subset=["norm_text","label"]).reset_index(drop=True)

# 4. 分层+分组切分（5 折的单折相当于 80/20）
y = df["label"].values
groups = df["norm_text"].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, dev_idx = next(sgkf.split(df, y, groups))

tr, dv = df.iloc[train_idx].copy(), df.iloc[dev_idx].copy()
print("New sizes:", len(tr), len(dv))

# 5. 断言无泄漏
overlap = set(tr["norm_text"]) & set(dv["norm_text"])
print("Overlap after split:", len(overlap))  # 必须为 0

# 6. 保存
tr[["text","label"]].to_csv("data/train_strat.csv", index=False)
dv[["text","label"]].to_csv("data/dev_strat.csv", index=False)
