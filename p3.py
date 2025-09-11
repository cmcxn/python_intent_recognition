import pandas as pd
from difflib import SequenceMatcher

train_df = pd.read_csv("data/train_strat.csv")
dev_df   = pd.read_csv("data/dev_strat.csv")

t = train_df["text"].astype(str).str.strip()
d = dev_df["text"].astype(str).str.strip()

# 精确重复
exact_overlap = set(t) & set(d)
print("Exact overlap:", len(exact_overlap))  # 必须为 0

# 近重复（抽样）
near = 0
for x in d.sample(min(len(d), 200), random_state=42):
    if any(SequenceMatcher(None, x, y).ratio() >= 0.95 for y in t.sample(min(len(t), 200), random_state=0)):
        near += 1
print("Near-dup (sampled):", near)  # 目标尽量接近
