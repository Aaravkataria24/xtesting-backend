import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# List of all usernames to process
USERNAMES = [
    "UseUniversalX", "TABASCOweb3", "vaibhavchellani", "intern", "0xMert_", 
    "cryptolyxe", "blknoiz06", "MustStopMurad", "gianinaskarlett", "frankdegods", 
    "notthreadguy", "_TJRTrades", "0xNairolf", "rajgokal", "lukebelmar", 
    "muststopNlG", "VitalikButerin", "TimBeiko", "mauritsneo", "aashatwt", 
    "param_eth", "yashvikram30", "okaykito", "_soulninja", "theunipcs", 
    "cz_binance", "TheCryptoLark", "JupiterExchange", "weremeow", "SOCKETProtocol", 
    "litocoen", "3orovik", "aeyakovenko", "lrettig", "musalbas", "jon_charb", 
    "avsa", "adamscochran", "koeppelmann", "0xCygaar", "cryptunez", "BullyEsq", 
    "solana", "phantom", "ethereum", "SuhailKakar", "IshitaaPandey", "ri5hitripathi"
]

def load_user_data(username):
    """Load data for a specific user"""
    filename = f"tweets_user_{username}.csv"
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded {filename} with {len(df)} rows")
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"⚠️ File not found or empty: {filename}, skipping...")
        return None

def apply_log_transform(df, columns):
    """Apply log1p transform to specified columns"""
    for col in columns:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])
    return df

def main():
    # Find the most recent scraping date to filter out recent tweets
    most_recent_date = None
    for username in USERNAMES:
        filename = f"tweets_user_{username}.csv"
        if os.path.exists(filename):
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(filename))
            if most_recent_date is None or file_modified_time > most_recent_date:
                most_recent_date = file_modified_time
    
    if most_recent_date is None:
        print("❌ No CSV files found for any users")
        return
    
    cutoff_date = most_recent_date - timedelta(days=2)  # 48 hours before scraping
    print(f"🕒 Using cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and combine all user data
    all_dfs = []
    for username in USERNAMES:
        df = load_user_data(username)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ No data loaded. Check filenames.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 Combined {len(combined_df)} tweets from all users")
    
    # Clean and preprocess the data
    
    # 1. Drop rows with missing essential values
    combined_df = combined_df.dropna(subset=["content", "likes", "retweets", "replies"])
    print(f"📊 {len(combined_df)} tweets after dropping missing values")
    
    # 2. Filter out tweets posted within 48 hours of scraping
    if "date_posted" in combined_df.columns and "time_posted" in combined_df.columns:
        combined_df["datetime"] = pd.to_datetime(
            combined_df["date_posted"] + " " + combined_df["time_posted"], 
            errors="coerce"
        )
        combined_df = combined_df[combined_df["datetime"] < cutoff_date]
        print(f"📊 {len(combined_df)} tweets after removing recent tweets")
    else:
        print("⚠️ No date/time columns found, skipping recent tweet filtering")
    
    # 3. Remove duplicates by tweet content
    initial_duplicates = combined_df.duplicated(subset=["content"]).sum()
    print(f"📊 Found {initial_duplicates} duplicate tweets")
    combined_df = combined_df.drop_duplicates(subset=["content"])
    print(f"📊 {len(combined_df)} tweets after removing duplicates")
    
    # 4. Apply log transforms to numeric engagement metrics
    numeric_columns = [
        "likes", "retweets", "replies", "follower_count", "view_count", "length"
    ]
    combined_df = apply_log_transform(combined_df, numeric_columns)
    
    # 5. Convert boolean-like string columns to actual boolean values
    bool_columns = [
        "has_image", "has_video", "has_link", "has_mention", 
        "has_crypto_mention", "is_quoting", "has_poll"
    ]
    for col in bool_columns:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].map({"yes": 1, "no": 0})
    
    # 6. Keep only the columns we need for fine-tuning
    # Remove only date_posted from the required columns, keep time_posted
    required_columns = [
        "content", 
        "likes_log", "retweets_log", "replies_log",
        "follower_count_log", "view_count_log", "length_log",
        "has_image", "has_video", "has_link", "has_mention", 
        "has_crypto_mention", "is_quoting", "has_poll",
        "time_posted"  # Explicitly include time_posted
    ]
    
    # Explicitly remove date_posted if it exists in the dataframe
    if "date_posted" in combined_df.columns:
        combined_df = combined_df.drop(columns=["date_posted"])
        print("📊 Removed date_posted column from the dataset")
    
    # And drop the datetime column we created for filtering
    if "datetime" in combined_df.columns:
        combined_df = combined_df.drop(columns=["datetime"])
        print("📊 Removed datetime column from the dataset")
    
    final_columns = [col for col in required_columns if col in combined_df.columns]
    
    # Check if we're missing any required columns
    missing_columns = set(required_columns) - set(final_columns)
    if missing_columns:
        print(f"⚠️ Missing some columns in the dataset: {missing_columns}")
    
    final_df = combined_df[final_columns]
    
    # 7. Drop any rows with NaN values in the final dataset
    rows_before = len(final_df)
    final_df = final_df.dropna()
    print(f"📊 Dropped {rows_before - len(final_df)} rows with NaN values")
    print(f"📊 Final dataset has {len(final_df)} tweets")
    
    # Save the processed dataset
    final_df.to_csv("processed_tweets_roberta.csv", index=False)
    print("✅ Preprocessing complete. Saved as 'processed_tweets_roberta.csv'")

if __name__ == "__main__":
    main() 