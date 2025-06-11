import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import json
import os

# Define the model architecture here instead of importing
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

class EnhancedRobertaRegressor(torch.nn.Module):
    def __init__(self, num_features=0, dropout_rate=0.2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        hidden_size = self.roberta.config.hidden_size
        self.num_features = num_features
        
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout_rate)
        
        if num_features > 0:
            self.feature_encoder = torch.nn.Sequential(
                torch.nn.Linear(num_features, 64),
                torch.nn.LayerNorm(64),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(64, 128),
                torch.nn.LayerNorm(128),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout_rate)
            )
            self.combined_layer = torch.nn.Linear(hidden_size + 128, hidden_size)
        
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense3 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.dense4 = torch.nn.Linear(hidden_size // 2, 3)
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_size)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        attention_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        pooled_output = attention_output.mean(dim=1)
        
        if features is not None and self.num_features > 0:
            feature_encoding = self.feature_encoder(features)
            combined_output = torch.cat([pooled_output, feature_encoding], dim=1)
            pooled_output = self.combined_layer(combined_output)
        
        x = self.layer_norm1(pooled_output)
        x = self.dropout(torch.nn.functional.gelu(self.dense1(x))) + pooled_output
        
        x = self.layer_norm2(x)
        x = self.dropout(torch.nn.functional.gelu(self.dense2(x))) + x
        
        x = self.dropout(torch.nn.functional.gelu(self.dense3(x)))
        x = self.dense4(x)
        
        return x

def analyze_error_distribution():
    # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load normalization parameters
    with open('normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    target_means = np.array(norm_params['target_means'])
    target_stds = np.array(norm_params['target_stds'])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
    
    # Load validation data
    print("Loading data...")
    df = pd.read_csv("processed_tweets_roberta.csv")
    texts = df["content"].tolist()
    targets = df[["likes_log", "retweets_log", "replies_log"]].values
    
    # Normalize targets
    normalized_targets = (targets - target_means) / target_stds
    
    # Create validation dataset
    val_dataset = TweetDataset(texts, normalized_targets, tokenizer, 256)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    num_features = len(norm_params.get('feature_columns', []))
    model = EnhancedRobertaRegressor(num_features=num_features).to(device)
    
    # Load model weights
    print("Loading model weights...")
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    
    # Handle state dict mismatch
    state_dict = checkpoint['model_state_dict']
    # Remove position_ids from state dict if it exists
    if 'roberta.embeddings.position_ids' in state_dict:
        del state_dict['roberta.embeddings.position_ids']
    
    # Load state dict with strict=False to ignore missing keys
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Collect predictions and actual values
    print("Running predictions...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    pred_array = np.array(all_preds)
    label_array = np.array(all_labels)
    
    # Denormalize predictions and labels
    pred_denorm = pred_array * target_stds + target_means
    label_denorm = label_array * target_stds + target_means
    
    # Calculate percentage error
    percent_error = np.abs((pred_denorm - label_denorm) / label_denorm) * 100
    
    # Define error buckets
    buckets = {
        "0-5%": 0,
        "5-10%": 0,
        "10-20%": 0,
        "20-50%": 0,
        "50-100%": 0,
        ">100%": 0
    }
    
    # Count predictions in each bucket
    for error in percent_error.flatten():
        if error <= 5:
            buckets["0-5%"] += 1
        elif error <= 10:
            buckets["5-10%"] += 1
        elif error <= 20:
            buckets["10-20%"] += 1
        elif error <= 50:
            buckets["20-50%"] += 1
        elif error <= 100:
            buckets["50-100%"] += 1
        else:
            buckets[">100%"] += 1
    
    # Calculate percentages
    total = sum(buckets.values())
    print("\nError Distribution:")
    print("------------------")
    for bucket, count in buckets.items():
        percentage = (count / total) * 100
        print(f"{bucket}: {percentage:.2f}% ({count} predictions)")
    
    # Calculate average error for predictions outside 5%
    outside_5_percent = percent_error[percent_error > 5]
    if len(outside_5_percent) > 0:
        avg_error = np.mean(outside_5_percent)
        median_error = np.median(outside_5_percent)
        print(f"\nFor predictions outside 5%:")
        print(f"Average error: {avg_error:.2f}%")
        print(f"Median error: {median_error:.2f}%")
        print(f"Max error: {np.max(outside_5_percent):.2f}%")
        print(f"Min error: {np.min(outside_5_percent):.2f}%")
    
    # Save detailed results
    results = {
        'error_distribution': buckets,
        'average_error_outside_5': float(np.mean(outside_5_percent)) if len(outside_5_percent) > 0 else None,
        'median_error_outside_5': float(np.median(outside_5_percent)) if len(outside_5_percent) > 0 else None,
        'max_error': float(np.max(percent_error)),
        'min_error': float(np.min(percent_error))
    }
    
    with open('error_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to error_analysis_results.json")

if __name__ == "__main__":
    analyze_error_distribution() 