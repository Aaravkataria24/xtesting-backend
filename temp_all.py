import pandas as pd
import os

USERNAMES = [
    "UseUniversalX", "TABASCOweb3", "vaibhavchellani", "intern", "0xMert_", "cryptolyxe", "blknoiz06", "MustStopMurad",
    "gianinaskarlett", "frankdegods", "notthreadguy", "_TJRTrades", "0xNairolf", "rajgokal", "lukebelmar", "muststopNlG",
    "VitalikButerin", "TimBeiko", "mauritsneo", "aashatwt", "param_eth", "yashvikram30", "okaykito", "_soulninja",
    "theunipcs", "cz_binance", "TheCryptoLark", "JupiterExchange", "weremeow", "SOCKETProtocol", "litocoen", "3orovik",
    "aeyakovenko", "lrettig", "musalbas", "jon_charb", "avsa", "adamscochran", "koeppelmann", "0xCygaar", "cryptunez",
    "BullyEsq", "solana", "phantom", "ethereum", "SuhailKakar", "IshitaaPandey", "ri5hitripathi"
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
    merged.to_csv("temp_all.csv", index=False)
    print(f"\nMerged {len(all_dfs)} files into temp_all.csv ({len(merged)} rows total)")
else:
    print("No files found to merge.")