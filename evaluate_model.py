import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Constants
MODEL_DIR = Path("finetuned_twitter_roberta_multi")  # Use Path to avoid Hugging Face repo ID issues
CSV_PATH = "processed_tweets_multi.csv"
REGRESSOR_PATH = "regressor_model.pkl"

# Load dataset
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR)
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Embedding generation using [CLS] token
def get_embeddings(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# Generate embeddings for validation set
print("🔍 Generating embeddings for validation set...")
val_embeddings = get_embeddings(val_df["content"])

# Load trained regressor
regressor = joblib.load(REGRESSOR_PATH)

# Predict engagement
y_true = val_df[["likes_log", "retweets_log", "replies_log"]].values
y_pred = regressor.predict(val_embeddings)

# Evaluate with RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))

# Print results
print("\n📊 Evaluation Results:")
print(f"Likes RMSE:    {rmse[0]:.4f}")
print(f"Retweets RMSE: {rmse[1]:.4f}")
print(f"Replies RMSE:  {rmse[2]:.4f}")
