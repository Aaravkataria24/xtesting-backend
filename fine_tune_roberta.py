import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Load processed data
df = pd.read_csv("processed_tweets_multi.csv")
texts = df["content"].tolist()
targets = df[["likes_log", "retweets_log", "replies_log"]].values

# Tokenizer and model
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Increased max_length to 256 to better handle longer tweets
MAX_LENGTH = 256

# Tokenize
encodings = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

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

# Train/val split
train_texts, val_texts, train_targets, val_targets = train_test_split(
    texts, targets, test_size=0.1, random_state=42
)

train_enc = tokenizer(train_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
val_enc = tokenizer(val_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

train_dataset = TweetDataset(train_enc, train_targets)
val_dataset = TweetDataset(val_enc, val_targets)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Custom model for multi-target regression with increased dropout
class RobertaRegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 3)  # 3 outputs

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        x = self.dropout(cls_output)
        return self.regressor(x)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaRegressionHead().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
loss_fn = nn.MSELoss()

# Training parameters
EPOCHS = 7
BEST_VAL_LOSS = float('inf')
PATIENCE = 3
patience_counter = 0
MAX_GRAD_NORM = 1.0  # Gradient clipping

# Create directory for model checkpoints
os.makedirs("checkpoints", exist_ok=True)

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0
    train_steps = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        total_train_loss += loss.item()
        train_steps += 1

    avg_train_loss = total_train_loss / train_steps
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    val_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
            val_steps += 1

    avg_val_loss = total_val_loss / val_steps
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    
    # Model checkpointing
    if avg_val_loss < BEST_VAL_LOSS:
        BEST_VAL_LOSS = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("✅ Saved new best model!")
    else:
        patience_counter += 1
        print(f"⚠️ No improvement for {patience_counter} epochs")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print("🛑 Early stopping triggered!")
        break

# Load best model for final save
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model_path = "finetuned_twitter_roberta_multi.pt"
torch.save(model.state_dict(), model_path)
print(f"✅ Saved final fine-tuned model to {model_path}")
