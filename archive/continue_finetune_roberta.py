# IF YOU WANT MORE EPOCHS


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
CHECKPOINT_PATH = "finetuned_twitter_roberta_multi.pt"
CSV_PATH = "processed_tweets_multi.csv"
EPOCHS_TO_RUN = 1  # just 1 more epoch

# Load data
df = pd.read_csv(CSV_PATH)
texts = df["content"].tolist()
targets = df[["likes_log", "retweets_log", "replies_log"]].values

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.targets[idx]
        return item

# Split & encode
train_texts, _, train_targets, _ = train_test_split(texts, targets, test_size=0.1, random_state=42)
train_enc = tokenizer(train_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
train_dataset = TweetDataset(train_enc, train_targets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model class
class RobertaRegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.regressor(x)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + checkpoint
model = RobertaRegressionHead().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

# Optimizer & loss
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

# 🔁 Continue training for 1 more epoch
for epoch in range(EPOCHS_TO_RUN):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Extra Epoch {epoch + 4} | Train loss: {total_loss / len(train_loader):.4f}")

# Save again
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"✅ Updated model saved to {CHECKPOINT_PATH}")
