import pandas as pd
import json
import os

# Ensure output directory exists
os.makedirs("splits", exist_ok=True)

# File paths for each data split
splits = {
    'train': 'SaladBench/train.parquet',
    'val': 'SaladBench/validation.parquet',
    'test': 'SaladBench/test.parquet'
}

# Format rows using the actual category_third value
def to_json_format(df):
    return [
        {
            "instruction": row["prompt"],
            "category": str(row["category_third"]).strip().replace(" ", "_"),
            "source": row["source"]
        }
        for _, row in df.iterrows()
    ]

# Step 1: Load train split first to determine valid categories
train_df = pd.read_parquet("hf://datasets/ncrispino/sc-jul-19-25/" + splits['train'])
train_formatted = to_json_format(train_df)

# Count prompts per category
category_counts = pd.Series([ex["category"] for ex in train_formatted]).value_counts()

# Keep only categories with >= 30 prompts in train
valid_categories = set(category_counts[category_counts >= 30].index)

print(f"Valid categories ({len(valid_categories)}): {sorted(valid_categories)}")

# Step 2: Process each split, filtering categories
for split_name, path in splits.items():
    df = pd.read_parquet("hf://datasets/ncrispino/sc-jul-19-25/" + path)
    formatted = to_json_format(df)

    # Filter to valid categories only
    filtered = [ex for ex in formatted if ex["category"] in valid_categories]

    out_path = f"splits/saladbench_harmful_{split_name}.json"
    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=4)

    print(f"✅ Saved {out_path} ({len(filtered)} examples after filtering)")


# ----------------------------
# JailbreakV-28k processing
# ----------------------------
import pandas as pd
import numpy as np

def format_jailbreakv(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure needed columns exist
    required = {"redteam_query", "policy"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Clean & dedupe by redteam_query
    df = df.copy()
    df["redteam_query"] = df["redteam_query"].astype(str).str.strip()
    df = df[df["redteam_query"].ne("") & df["redteam_query"].notna()]
    df = df.drop_duplicates(subset=["redteam_query"], keep="first")

    # Build JSON-ready DataFrame
    return pd.DataFrame({
        "instruction": df["redteam_query"],
        "category": df["policy"].astype(str).str.strip().str.replace(" ", "_"),
        "source": "JailBreakV"
    })

# Load and format
splits = {
    'mini_JailBreakV_28K': 'JailBreakV_28K/mini_JailBreakV_28K.csv',
    'JailBreakV_28K': 'JailBreakV_28K/JailBreakV_28K.csv'
}
jb_df = pd.read_csv("hf://datasets/JailbreakV-28K/JailBreakV-28k/" + splits["JailBreakV_28K"])
jb_formatted = format_jailbreakv(jb_df)

# Reproducible RNG
rng = np.random.default_rng(42)

def split_counts(k, ratios=(0.4, 0.1, 0.5)):
    raw = np.array(ratios) * k
    base = np.floor(raw).astype(int)
    rem = raw - base
    need = k - base.sum()
    order = np.argsort(-rem)
    for j in range(need):
        base[order[j % len(base)]] += 1
    return tuple(base.tolist())

train_rows, val_rows, test_rows = [], [], []

# >>> FIX: stratify on jb_formatted (has 'category'), not jb_df
for cat, group in jb_formatted.groupby("category"):
    print(cat, len(group))
    idxs = group.index.to_numpy()
    rng.shuffle(idxs)

    n_train, n_val, n_test = split_counts(len(idxs), (0.4, 0.1, 0.5))
    train_rows.extend(idxs[:n_train])
    val_rows.extend(idxs[n_train:n_train + n_val])
    test_rows.extend(idxs[n_train + n_val:])

train_df = jb_formatted.loc[train_rows].reset_index(drop=True)
val_df   = jb_formatted.loc[val_rows].reset_index(drop=True)
test_df  = jb_formatted.loc[test_rows].reset_index(drop=True)

# Save as JSON
os.makedirs("splits", exist_ok=True)
def save_df(df, name):
    path = f"splits/jailbreakv_harmful_{name}.json"
    df.to_json(path, orient="records", indent=2, force_ascii=False)
    print(f"✅ Saved {path} ({len(df)} examples)")

save_df(train_df, "train")
save_df(val_df, "val")
save_df(test_df, "test")

# ----------------------------
# StrongREJECT processing
# ----------------------------
import pandas as pd

def format_strongreject(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure needed columns exist
    required = {"prompt", "category"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Clean & dedupe by prompt
    df = df.copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].ne("") & df["prompt"].notna()]
    df = df.drop_duplicates(subset=["prompt"], keep="first")

    # Build JSON-ready DataFrame
    return pd.DataFrame({
        "instruction": df["prompt"],
        "category": df["category"].astype(str).str.strip().str.replace(" ", "_"),
        "source": "StrongREJECT"
    })

# Load dataset (full CSV assumed)
sr_df = pd.read_parquet("hf://datasets/walledai/StrongREJECT/data/train-00000-of-00001.parquet")
sr_formatted = format_strongreject(sr_df)

# Deterministic split (40/10/50 stratified by category)
rng = np.random.default_rng(42)
train_rows, val_rows, test_rows = [], [], []

for cat, group in sr_formatted.groupby("category"):
    print(cat, len(group))
    idxs = group.index.to_numpy()
    rng.shuffle(idxs)

    n_train, n_val, n_test = split_counts(len(idxs), (0.4, 0.1, 0.5))
    train_rows.extend(idxs[:n_train])
    val_rows.extend(idxs[n_train:n_train + n_val])
    test_rows.extend(idxs[n_train + n_val:])

train_df = sr_formatted.loc[train_rows].reset_index(drop=True)
val_df   = sr_formatted.loc[val_rows].reset_index(drop=True)
test_df  = sr_formatted.loc[test_rows].reset_index(drop=True)


def save_df(df, name):
    path = f"splits/strongreject_harmful_{name}.json"
    df.to_json(path, orient="records", indent=2, force_ascii=False)
    print(f"✅ Saved {path} ({len(df)} examples)")

# Save as JSON
save_df(train_df, "train")
save_df(val_df, "val")
save_df(test_df, "test")
