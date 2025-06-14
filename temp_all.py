import pandas as pd
import os

USERNAMES = [
    "TABASCOweb3", "intern", "HeetTike", "vaibhavchellani"
]

all_dfs = []
for username in USERNAMES:
    filename = f"tweets_user_{username}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        all_dfs.append(df)
        print(f"Loaded {filename} ({len(df)} rows)")
    else:
        print(f"File not found: {filename}")

if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv("temp_4.csv", index=False)
    print(f"\nMerged {len(all_dfs)} files into temp_all.csv ({len(merged)} rows total)")
else:
    print("No files found to merge.")