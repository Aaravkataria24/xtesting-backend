import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
CHECKPOINT_PATH = "finetuned_twitter_roberta_multi.pt"
REGRESSOR_PATH = "regressor_model.pkl"
CSV_PATH = "processed_tweets_multi.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)
train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

# Custom wrapper for regression head
import torch.nn as nn
class RobertaRegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = base_model
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.regressor(x)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaRegressionHead().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Embedding generator
def get_embeddings(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# Generate embeddings for training set
print("🔍 Generating embeddings for training set...")
train_embeddings = get_embeddings(train_df["content"])
train_targets = train_df[["likes_log", "retweets_log", "replies_log"]].values

# Train the regression model
print("🧠 Training regression model...")
regressor = MultiOutputRegressor(GradientBoostingRegressor())
regressor.fit(train_embeddings, train_targets)

# Save the trained regressor
joblib.dump(regressor, REGRESSOR_PATH)
print(f"✅ Saved regressor model to '{REGRESSOR_PATH}'")
