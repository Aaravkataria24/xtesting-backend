import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import os
import gc
import json
import torch.nn.functional as F
import signal
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

def save_checkpoint(state_dict, filename, is_best=False):
    """Helper function to save checkpoints with error handling"""
    try:
        os.makedirs('checkpoints', exist_ok=True)
        filepath = os.path.join('checkpoints', filename)
        torch.save(state_dict, filepath)
        print(f"✅ Successfully saved {'best model' if is_best else 'checkpoint'} to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving {'best model' if is_best else 'checkpoint'} to {filename}: {str(e)}")
        return False

def cleanup_old_checkpoints():
    """Remove all checkpoints except best_model.pt"""
    try:
        os.makedirs('checkpoints', exist_ok=True)
        for file in os.listdir('checkpoints'):
            if file != "best_model.pt":
                os.remove(os.path.join('checkpoints', file))
        print("✅ Cleaned up old checkpoints")
    except Exception as e:
        print(f"❌ Error cleaning up checkpoints: {str(e)}")

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
    def __init__(self, num_features=11, dropout_rate=0.2):  # Set num_features to match your features
        super().__init__()
        self.roberta = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        hidden_size = self.roberta.config.hidden_size  # This is 768 for RoBERTa base
        self.num_features = num_features
        
        # Feature extraction layers
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout_rate)
        
        # Feature processing if we have additional features
        if num_features > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
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
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        attention_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        pooled_output = attention_output.mean(dim=1)  # [batch_size, hidden_size]
        
        if features is not None and self.num_features > 0:
            feature_encoding = self.feature_encoder(features)
            combined_output = torch.cat([pooled_output, feature_encoding], dim=1)
            pooled_output = self.combined_layer(combined_output)
        
        x = self.layer_norm1(pooled_output)
        x = self.dropout(F.gelu(self.dense1(x))) + pooled_output  # Residual connection
        x = self.layer_norm2(x)
        x = self.dropout(F.gelu(self.dense2(x))) + x  # Residual connection
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
    ape = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
    error_margins = [5, 10, 20, 30, 50, 100]
    margin_distribution = []
    for margin in error_margins:
        within_margin = np.mean(ape <= margin) * 100
        margin_distribution.append(within_margin)
    median_ape = np.median(ape)
    mean_ape = np.mean(ape)
    r2 = stats.pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
    print(f"\n📊 {metric_name} Error Analysis:")
    print(f"Median Absolute Percentage Error: {median_ape:.2f}%")
    print(f"Mean Absolute Percentage Error: {mean_ape:.2f}%")
    print(f"Pearson R: {r2:.4f}")
    print("\nError Margin Distribution:")
    for margin, percentage in zip(error_margins, margin_distribution):
        print(f"Within {margin}% error: {percentage:.2f}% of predictions")
    plt.figure(figsize=(10, 6))
    sns.histplot(ape, bins=50, kde=True)
    plt.axvline(median_ape, color='r', linestyle='--', label=f'Median Error: {median_ape:.2f}%')
    plt.title(f'{metric_name} - Error Distribution')
    plt.xlabel('Absolute Percentage Error (%)')
    plt.ylabel('Count')
    plt.legend()
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig(f'evaluation_results/{metric_name.lower()}_error_distribution.png')
    plt.close()
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
    metrics = {}
    for i, metric_name in enumerate(metric_names):
        metrics[metric_name] = calculate_error_metrics(
            all_targets[:, i],
            all_predictions[:, i],
            metric_name
        )
    return metrics

def train_model():
    # This function will be called from your notebook, not from inside this file
    pass  # The actual implementation will use variables defined in your notebook 