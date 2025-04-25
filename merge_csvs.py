import pandas as pd

usernames = ["TheCryptoLark", "JupiterExchange", "weremeow", "SOCKETProtocol", "litocoen", "3orovik", "aeyakovenko", "lrettig", "musalbas", "jon_charb", "avsa", "adamscochran", "koeppelmann", "0xCygaar", "cryptunez", "BullyEsq", "solana", "phantom", "ethereum"]

search_queries = ["chain%20abstraction", "interop", "rollup", "solana", "trenches", "multi-chain", "dApp", "onchain", "web3", "defi", "nft", "gamefi", "socialfi", "dao", "wallet", "staking", "bridging", "L2"]

# First, try to load the existing tweets_all_users.csv
try:
    existing_df = pd.read_csv("tweets_all_users.csv")
    print(f"✅ Loaded existing tweets_all_users.csv with {len(existing_df)} rows")
except FileNotFoundError:
    print("⚠️ No existing tweets_all_users.csv found. Creating new file.")
    existing_df = pd.DataFrame()

all_dfs = []

# Process username files
for username in usernames:
    file_name = f"tweets_{username}.csv"
    try:
        df = pd.read_csv(file_name)
        if not df.empty:
            all_dfs.append(df)
            print(f"✅ Loaded {file_name} with {len(df)} rows")
        else:
            print(f"⚠️ Empty file: {file_name}, skipping...")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"⚠️ File not found or empty: {file_name}, skipping...")

# Process search query files
for search_query in search_queries:
    file_name = f"tweets_search_query_{search_query}.csv"
    try:
        df = pd.read_csv(file_name)
        if not df.empty:
            all_dfs.append(df)
            print(f"✅ Loaded {file_name} with {len(df)} rows")
        else:
            print(f"⚠️ Empty file: {file_name}, skipping...")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"⚠️ File not found or empty: {file_name}, skipping...")

# Combine all new dataframes
if all_dfs:
    new_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates based on content
    if not existing_df.empty:
        # Get unique tweets from new data that don't exist in the old data
        unique_new_tweets = new_df[~new_df['content'].isin(existing_df['content'])]
        print(f"\n📊 Found {len(unique_new_tweets)} new unique tweets to add")
        
        # Combine existing and new unique tweets
        combined_df = pd.concat([existing_df, unique_new_tweets], ignore_index=True)
    else:
        combined_df = new_df
    
    # Save the combined result
    combined_df.to_csv("tweets_all_users.csv", index=False)
    print(f"\n✅ Combined CSV saved as tweets_all_users.csv with {len(combined_df)} rows")
else:
    print("❌ No new files loaded. Check filenames.")
