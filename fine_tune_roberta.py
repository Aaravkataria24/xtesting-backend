# add automation so that if it begins overfitting (val loss starts increasing), it stops and saves the best model
# if it already saves best model only, then no need of above step

# make sure there are checkpoints after each epoch

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
import signal
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

# Global variable to track if training should stop
should_stop_training = False

def signal_handler(signum, frame):
    global should_stop_training
    print("\n⚠️ Training interruption requested. Saving best model and stopping...")
    should_stop_training = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Set up paths in Google Drive
DRIVE_PATH = "/content/drive/MyDrive/tweetlab_model"
CHECKPOINT_PATH = os.path.join(DRIVE_PATH, "checkpoints")
LOG_PATH = os.path.join(DRIVE_PATH, "logs")

# Create directories at the start
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Checkpoint path: {CHECKPOINT_PATH}")
print(f"Log path: {LOG_PATH}")

def save_checkpoint(state_dict, filename, is_best=False):
    """Helper function to save checkpoints with error handling"""
    try:
        filepath = os.path.join(CHECKPOINT_PATH, filename)
        torch.save(state_dict, filepath)
        print(f"✅ Successfully saved {'best model' if is_best else 'checkpoint'} to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving {'best model' if is_best else 'checkpoint'} to {filename}: {str(e)}")
        return False

def cleanup_old_checkpoints():
    """Remove all checkpoints except best_model.pt"""
    try:
        for file in os.listdir(CHECKPOINT_PATH):
            if file != "best_model.pt":
                os.remove(os.path.join(CHECKPOINT_PATH, file))
        print("✅ Cleaned up old checkpoints")
    except Exception as e:
        print(f"❌ Error cleaning up checkpoints: {str(e)}")

# Load and preprocess data
df = pd.read_csv("processed_tweets_roberta.csv")
print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Identify text and numeric columns
text_column = "content"
numeric_columns = [col for col in df.columns if col != text_column]
target_columns = ["likes_log", "retweets_log", "replies_log"]

# Handle time_posted column specially
time_features = []
if 'time_posted' in df.columns:
    # Convert time_posted to numeric features
    print("Converting time_posted to numeric features...")
    # Extract hour and minute as separate features
    try:
        df['hour'] = df['time_posted'].apply(lambda x: int(str(x).split(':')[0]))
        df['minute'] = df['time_posted'].apply(lambda x: int(str(x).split(':')[1]))
        time_features = ['hour', 'minute']
        # Remove original time_posted column from feature columns
        numeric_columns.remove('time_posted')
        numeric_columns.extend(time_features)
    except Exception as e:
        print(f"Error processing time_posted: {e}")
        if 'time_posted' in numeric_columns:
            numeric_columns.remove('time_posted')

# Remove view_count_log from feature columns
feature_columns = [col for col in numeric_columns if col not in target_columns and col != 'view_count_log']

print(f"Text column: {text_column}")
print(f"Target columns: {target_columns}")
print(f"Feature columns (excluding view_count_log): {feature_columns}")
print(f"Number of features: {len(feature_columns)}")  # Should be 11

# Print sample data to verify structure
print("\nSample data row:")
print(df.iloc[0][feature_columns].to_dict())

texts = df[text_column].tolist()
targets = df[target_columns].values
features = df[feature_columns].values if feature_columns else None

# Calculate mean and std for normalization of targets
target_means = np.mean(targets, axis=0)
target_stds = np.std(targets, axis=0)
normalized_targets = (targets - target_means) / target_stds

# If we have numeric features, normalize them too
if features is not None:
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    # Handle zero standard deviation (constant features)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)
    normalized_features = (features - feature_means) / feature_stds
    
    # Save normalization parameters for both targets and features
    normalization_params = {
        'target_means': target_means.tolist(),
        'target_stds': target_stds.tolist(),
        'feature_means': feature_means.tolist(),
        'feature_stds': feature_stds.tolist(),
        'target_columns': target_columns,
        'feature_columns': feature_columns
    }
else:
    # Save normalization parameters only for targets
    normalization_params = {
        'target_means': target_means.tolist(),
        'target_stds': target_stds.tolist(),
        'target_columns': target_columns
    }

# Save normalization parameters
with open('normalization_params.json', 'w') as f:
    json.dump(normalization_params, f)
    print("✅ Saved normalization parameters to normalization_params.json")

# Model configuration
# can change batch_size so that training is faster (note, will consume more memory)
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 2  # Just 2 epochs as requested
PATIENCE = 5
WARMUP_STEPS = 100  # Reduced for shorter training
MAX_GRAD_NORM = 1.0

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Enhanced Dataset class with numeric features
class TweetDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length, features=None):
        self.encodings = tokenizer(texts, padding=True, truncation=True, 
                                 max_length=max_length, return_tensors="pt")
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.features = torch.tensor(features, dtype=torch.float32) if features is not None else None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.targets[idx]
        if self.features is not None:
            item["features"] = self.features[idx]
        return item

# Advanced model architecture with numeric features
class EnhancedRobertaRegressor(nn.Module):
    def __init__(self, num_features=11, dropout_rate=0.2):  # Fixed to 11 features
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size  # This is 768 for RoBERTa base
        self.num_features = num_features  # Should be 11
        
        # Feature extraction layers
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout_rate)
        
        # Feature processing if we have additional features
        if num_features > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(11, 64),  # Fixed to 11 features
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            
            # Combined processing of text and numeric features
            self.combined_layer = nn.Linear(hidden_size + 128, hidden_size)
        
        # Multiple dense layers with residual connections
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense4 = nn.Linear(hidden_size // 2, 3)  # Direct output to 3 targets
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids, attention_mask, features=None):
        # Get RoBERTa embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply self-attention
        attention_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        pooled_output = attention_output.mean(dim=1)  # [batch_size, hidden_size]
        
        # Incorporate numeric features if available
        if features is not None and self.num_features > 0:
            feature_encoding = self.feature_encoder(features)
            combined_output = torch.cat([pooled_output, feature_encoding], dim=1)
            pooled_output = self.combined_layer(combined_output)
        
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

def calculate_error_metrics(y_true, y_pred, metric_name=""):
    """Calculate detailed error metrics including error margin distributions"""
    # Calculate absolute percentage errors
    ape = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100  # Add small epsilon to avoid division by zero
    
    # Calculate error margins
    error_margins = [5, 10, 20, 30, 50, 100]  # Error margins in percentage
    margin_distribution = []
    
    # Calculate percentage of predictions within each error margin
    for margin in error_margins:
        within_margin = np.mean(ape <= margin) * 100
        margin_distribution.append(within_margin)
    
    # Calculate median and mean absolute percentage error
    median_ape = np.median(ape)
    mean_ape = np.mean(ape)
    
    # Calculate R² score
    r2 = stats.r2_score(y_true, y_pred)
    
    # Print detailed metrics
    print(f"\n📊 {metric_name} Error Analysis:")
    print(f"Median Absolute Percentage Error: {median_ape:.2f}%")
    print(f"Mean Absolute Percentage Error: {mean_ape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print("\nError Margin Distribution:")
    for margin, percentage in zip(error_margins, margin_distribution):
        print(f"Within {margin}% error: {percentage:.2f}% of predictions")
    
    # Create error distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(ape, bins=50, kde=True)
    plt.axvline(median_ape, color='r', linestyle='--', label=f'Median Error: {median_ape:.2f}%')
    plt.title(f'{metric_name} - Error Distribution')
    plt.xlabel('Absolute Percentage Error (%)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'evaluation_results/{metric_name.lower()}_error_distribution.png')
    plt.close()
    
    # Create scatter plot of predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'{metric_name} - Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig(f'evaluation_results/{metric_name.lower()}_predictions_vs_actual.png')
    plt.close()
    
    return {
        'median_ape': median_ape,
        'mean_ape': mean_ape,
        'r2': r2,
        'margin_distribution': dict(zip(error_margins, margin_distribution))
    }

def evaluate_model(model, val_loader, device, metric_names):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].cpu().numpy()
            
            if 'features' in batch:
                features = batch['features'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = outputs.cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Denormalize predictions and targets
    target_means = np.array(normalization_params['target_means'])
    target_stds = np.array(normalization_params['target_stds'])
    
    denormalized_predictions = np.exp(all_predictions * target_stds + target_means) - 1
    denormalized_targets = np.exp(all_targets * target_stds + target_means) - 1
    
    # Calculate metrics for each target
    metrics = {}
    for i, metric_name in enumerate(metric_names):
        metrics[metric_name] = calculate_error_metrics(
            denormalized_targets[:, i],
            denormalized_predictions[:, i],
            metric_name
        )
    
    return metrics

def train_model():
    """Main training function with proper structure and error handling"""
    try:
        # 1. Data Preparation
        print("📊 Preparing data...")
        train_texts, val_texts, train_targets, val_targets = train_test_split(
            texts, normalized_targets, test_size=0.1, random_state=42
        )
        
        if features is not None:
            train_features, val_features = train_test_split(
                normalized_features, test_size=0.1, random_state=42
            )
        else:
            train_features = val_features = None
        
        # Create datasets and dataloaders
        train_dataset = TweetDataset(train_texts, train_targets, tokenizer, MAX_LENGTH, train_features)
        val_dataset = TweetDataset(val_texts, val_targets, tokenizer, MAX_LENGTH, val_features)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        print(f"✅ Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # 2. Model Setup
        print("\n🤖 Setting up model...")
        model = EnhancedRobertaRegressor(num_features=11).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
        )
        loss_fn = CombinedLoss()
        
        # Create evaluation directory
        os.makedirs('evaluation_results', exist_ok=True)
        cleanup_old_checkpoints()
        
        print("✅ Model setup complete")
        
        # 3. Training State
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        training_history = []
        
        # 4. Training Loop
        print("\n🚀 Starting training...")
        for epoch in range(EPOCHS):
            if should_stop_training:
                print("\n⚠️ Training interrupted by user")
                break
            
            epoch_start_time = time.time()
            
            # 4.1 Training Phase
            model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Train]')
            
            for batch in train_pbar:
                if should_stop_training:
                    break
                
                # Forward pass
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                if "features" in batch:
                    features_batch = batch["features"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features_batch)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss and backprop
                loss = loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Memory cleanup
                del outputs, loss
                torch.cuda.empty_cache()
                gc.collect()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # 4.2 Validation Phase
            model.eval()
            val_loss = 0
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Val]')
            
            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    if "features" in batch:
                        features_batch = batch["features"].to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features_batch)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': f'{val_loss/len(val_loader):.4f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            
            # 4.3 Epoch Summary
            epoch_time = time.time() - epoch_start_time
            print(f"\n📊 Epoch {epoch + 1}/{EPOCHS} Summary:")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # 4.4 Detailed Evaluation
            print("\n🔍 Running detailed evaluation...")
            metrics = evaluate_model(model, val_loader, device, ['Likes', 'Retweets', 'Replies'])
            
            # Save metrics and checkpoint
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'time': epoch_time
            }
            training_history.append(epoch_data)
            
            with open(f'evaluation_results/epoch_{epoch+1}_metrics.json', 'w') as f:
                json.dump(epoch_data, f, indent=2)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'training_history': training_history
            }
            
            if save_checkpoint(checkpoint, f'checkpoint_epoch_{epoch+1}.pt'):
                print(f"✅ Saved checkpoint for epoch {epoch+1}")
            
            # 4.5 Model Selection
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                if save_checkpoint({'model_state_dict': best_model_state}, 'best_model.pt', is_best=True):
                    print("🏆 New best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
        
        # 5. Final Evaluation
        print("\n📈 Running final evaluation...")
        final_metrics = evaluate_model(model, val_loader, device, ['Likes', 'Retweets', 'Replies'])
        
        # Save final results
        final_results = {
            'final_metrics': final_metrics,
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1
        }
        
        with open('evaluation_results/final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n✨ Training completed successfully!")
        return model
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    train_model()
