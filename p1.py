import pandas as pd
from difflib import SequenceMatcher

train_df = pd.read_csv("data/train.csv")
dev_df   = pd.read_csv("data/test.csv")

t = train_df["text"].astype(str).str.strip()
d = dev_df["text"].astype(str).str.strip()

# 精确重复
exact_overlap = set(t) & set(d)
print("Exact overlap:", len(exact_overlap))

# 近重复（简单版）：字符相似度≥0.95
near = 0
for x in d.sample(min(len(d), 200), random_state=42):   # 子样本以提速
    if any(SequenceMatcher(None, x, y).ratio() >= 0.95 for y in t.sample(min(len(t), 200), random_state=0)):
        near += 1
print("Near-dup (sampled):", near)

print("Train label counts:\n", train_df["label"].value_counts())
print("Dev   label counts:\n", dev_df["label"].value_counts())

# from sklearn.model_selection import train_test_split
# df = pd.concat([train_df, dev_df], ignore_index=True)
# tr, dv = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
# tr.to_csv("data/train_strat.csv", index=False)
# dv.to_csv("data/dev_strat.csv", index=False)