import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("tweets_all_users.csv")

# Drop rows with missing content or engagement
df = df.dropna(subset=["content", "likes", "retweets", "replies"])

# Drop the 'username' column (not needed for training)
if "username" in df.columns:
    df = df.drop(columns=["username"])

# Remove duplicates by tweet content
df = df.drop_duplicates(subset=["content"])

# Remove tweets with all engagement metrics as 0
df = df[(df["likes"] > 0) | (df["retweets"] > 0) | (df["replies"] > 0)]

# Apply log1p transform (log(x + 1)) for better regression modeling
df["likes_log"] = np.log1p(df["likes"])
df["retweets_log"] = np.log1p(df["retweets"])
df["replies_log"] = np.log1p(df["replies"])

# Keep only the columns we need for training
final_df = df[["content", "likes_log", "retweets_log", "replies_log"]]

# Save to new CSV
final_df.to_csv("processed_tweets_multi.csv", index=False)

print("✅ Preprocessing complete. Saved as 'processed_tweets_multi.csv'")
