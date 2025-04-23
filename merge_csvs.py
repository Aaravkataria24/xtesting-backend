import pandas as pd

usernames = [
    "TABASCOweb3", "vaibhavchellani", "intern", "0xMert_", "cryptolyxe", "blknoiz06",
    "MustStopMurad", "gianinaskarlett", "frankdegods", "notthreadguy", "_TJRTrades",
    "0xNairolf", "rajgokal", "lukebelmar", "muststopnigg", "VitalikButerin",
    "TimBeiko", "gavofyork", "cz_binance"
]

all_dfs = []

for username in usernames:
    file_name = f"tweets_{username}.csv"
    try:
        df = pd.read_csv(file_name)
        all_dfs.append(df)
        print(f"✅ Loaded {file_name} with {len(df)} rows")
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_name}, skipping...")

# Combine all and save
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv("tweets_all_users.csv", index=False)
    print(f"\n✅ Combined CSV saved as tweets_all_users.csv with {len(combined_df)} rows")
else:
    print("❌ No files loaded. Check filenames.")
