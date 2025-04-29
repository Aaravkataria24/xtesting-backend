import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import os
import gc
import json
from datetime import datetime
import torch.nn.functional as F

# Load and preprocess data
df = pd.read_csv("processed_tweets_multi.csv")
texts = df["content"].tolist()
targets = df[["likes_log", "retweets_log", "replies_log"]].values

# Calculate mean and std for normalization
target_means = np.mean(targets, axis=0)
target_stds = np.std(targets, axis=0)
normalized_targets = (targets - target_means) / target_stds

# Save normalization parameters
normalization_params = {
    'means': target_means.tolist(),
    'stds': target_stds.tolist()
}
with open('normalization_params.json', 'w') as f:
    json.dump(normalization_params, f)

# Model configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 15
PATIENCE = 5
WARMUP_STEPS = 1000
MAX_GRAD_NORM = 1.0

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Enhanced Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.encodings = tokenizer(texts, padding=True, truncation=True, 
                                 max_length=max_length, return_tensors="pt")
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.targets[idx]
        return item

# Advanced model architecture
class EnhancedRobertaRegressor(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size  # This is 768 for RoBERTa base
        
        # Feature extraction layers
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout_rate)
        
        # Multiple dense layers with residual connections
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense4 = nn.Linear(hidden_size // 2, 3)  # Direct output to 3 targets
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids, attention_mask):
        # Get RoBERTa embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply self-attention
        attention_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        pooled_output = attention_output.mean(dim=1)  # [batch_size, hidden_size]
        
        # Dense layers with residual connections and layer normalization
        x = self.layer_norm1(pooled_output)
        x = self.dropout(F.gelu(self.dense1(x))) + pooled_output  # Residual connection
        
        x = self.layer_norm2(x)
        x = self.dropout(F.gelu(self.dense2(x))) + x  # Residual connection
        
        # Final layers without residual connections
        x = self.dropout(F.gelu(self.dense3(x)))
        x = self.dense4(x)
        
        return x

# Custom loss function combining MSE and Huber loss
class CombinedLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=delta)
        
    def forward(self, pred, target):
        return 0.7 * self.mse(pred, target) + 0.3 * self.huber(pred, target)

# Training setup
def train_model():
    # Split data
    train_texts, val_texts, train_targets, val_targets = train_test_split(
        texts, normalized_targets, test_size=0.1, random_state=42
    )

    # Create datasets
    train_dataset = TweetDataset(train_texts, train_targets, tokenizer, MAX_LENGTH)
    val_dataset = TweetDataset(val_texts, val_targets, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedRobertaRegressor().to(device)
    
    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    loss_fn = CombinedLoss()
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Training loop
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            
            # Training phase
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
                # Save checkpoint every 100 batches
                if batch_idx % 100 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, "checkpoints/latest_checkpoint.pt")
                
                # Clear memory
                del outputs, loss
                torch.cuda.empty_cache()
                gc.collect()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs, labels)
                    total_val_loss += loss.item()
                    
                    # Store predictions and labels for metrics
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    del outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Calculate percentage within 5% error
            pred_array = np.array(all_preds)
            label_array = np.array(all_labels)
            
            # Denormalize predictions and labels
            pred_denorm = pred_array * target_stds + target_means
            label_denorm = label_array * target_stds + target_means
            
            # Calculate percentage error
            percent_error = np.abs((pred_denorm - label_denorm) / label_denorm) * 100
            within_5_percent = np.mean(percent_error <= 5.0) * 100
            
            # Log metrics
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'within_5_percent': within_5_percent,
                'timestamp': datetime.now().isoformat()
            }
            
            with open("logs/training_log.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Within 5% error: {within_5_percent:.2f}%")
            
            # Model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_val_loss,
                    'within_5_percent': within_5_percent
                }, "checkpoints/best_model.pt")
                print("✅ Saved new best model!")
            else:
                patience_counter += 1
                print(f"⚠️ No improvement for {patience_counter} epochs")
            
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping triggered!")
                break
            
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"❌ Training interrupted: {str(e)}")
        raise e

    # Save final model
    model.load_state_dict(torch.load("checkpoints/best_model.pt")['model_state_dict'])
    torch.save(model.state_dict(), "finetuned_twitter_roberta_multi.pt")
    print("✅ Training complete!")

if __name__ == "__main__":
    train_model()
