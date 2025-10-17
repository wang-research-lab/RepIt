import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# Ensure output directory exists
os.makedirs("splits", exist_ok=True)

# SaladBench paths
salad_paths = {
    'train': 'SaladBench/train.parquet',
    'val': 'SaladBench/validation.parquet',
    'test': 'SaladBench/test.parquet'
}

# WMD CSV files (flat, unsplit)
wmd_categories = ['bio', 'chem', 'cyber']
wmd_csv_template = "rewritten_WMDP_{}.csv"

# Format function
def to_json_format_salad(df):
    return [
        {
            "instruction": row["prompt"],
            "category": str(row["category_secondary"]).strip().replace(" ", "_"),
            "source": row["source"]
        }
        for _, row in df.iterrows()
    ]

def to_json_format_wmd(df, wmd_cat):
    return [
        {
            "instruction": row["rewritten_prompt"],
            "category": wmd_cat,
            "source": "WMDP"
        }
        for _, row in df.iterrows()
    ]

# Load and process
for cat in wmd_categories:
    wmd_path = wmd_csv_template.format(cat)
    wmd_df = pd.read_csv(wmd_path)

    # Shuffle and split manually
    train_df, temp_df = train_test_split(wmd_df, train_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=5/6, random_state=42)

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for split_name, wmd_split_df in split_dfs.items():
    
        # Format WMD split
        wmd_data = to_json_format_wmd(wmd_split_df, f"wmdp_{cat}".upper())

        out_path = f"splits/wmd_{cat}_harmful_{split_name}.json"
        #with open(out_path, "w") as f:
        #    json.dump(wmd_data, f, indent=4)

        print(f"✅ Saved {out_path}, ({len(wmd_data)} examples)")
