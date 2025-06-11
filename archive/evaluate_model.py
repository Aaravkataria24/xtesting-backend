import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import gc

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
CHECKPOINT_PATH = "finetuned_twitter_roberta_multi.pt"
REGRESSOR_PATH = "regressor_model.pkl"
SCALER_PATH = "scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"
CSV_PATH = "processed_tweets_multi.csv"

# Load dataset
print("📊 Loading dataset...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset file not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
_, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Loaded {len(test_df)} test samples")

# Load models and scalers
print("🤖 Loading models...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load RoBERTa model: {str(e)}")

try:
    regressor = joblib.load(REGRESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load regressor model: {str(e)}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load feature scaler: {str(e)}")

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
print(f"Using device: {device}")

# Clear GPU memory if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

model = RobertaRegressionHead().to(device)

# Load state dict with strict=False to ignore missing keys
try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Successfully loaded model checkpoint")
except Exception as e:
    raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")

model.eval()

# Enhanced embedding generator with progress bar
def get_enhanced_embeddings(texts, batch_size=16):
    embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get [CLS] token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Get mean and max pooling of all tokens
            mean_pooling = outputs.last_hidden_state.mean(dim=1)
            max_pooling = outputs.last_hidden_state.max(dim=1)[0]
            
            # Concatenate all features
            combined_features = torch.cat([
                cls_embeddings,
                mean_pooling,
                max_pooling
            ], dim=1)
            
            embeddings.append(combined_features.cpu().numpy())
            
            # Clear memory
            del outputs, cls_embeddings, mean_pooling, max_pooling, combined_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return np.concatenate(embeddings, axis=0)

# Generate embeddings for test set
print("🔍 Generating embeddings for test set...")
test_embeddings = get_enhanced_embeddings(test_df["content"])
test_embeddings_scaled = scaler.transform(test_embeddings)

# Get predictions
print("📈 Making predictions...")
test_predictions = np.zeros((len(test_embeddings_scaled), 3))  # Initialize predictions array

# Make predictions for each target
targets = ['likes_log', 'retweets_log', 'replies_log']
for i, target in enumerate(targets):
    print(f"Predicting {target}...")
    test_predictions[:, i] = regressor[target].predict(test_embeddings_scaled)

test_targets = test_df[["likes_log", "retweets_log", "replies_log"]].values

# Convert log predictions to actual numbers
test_predictions_actual = np.exp(test_predictions)
test_targets_actual = np.exp(test_targets)

# Calculate metrics for both log and actual values
metrics = {}
for i, target in enumerate(['likes', 'retweets', 'replies']):
    metrics[target] = {
        'Log Scale': {
            'MSE': mean_squared_error(test_targets[:, i], test_predictions[:, i]),
            'RMSE': np.sqrt(mean_squared_error(test_targets[:, i], test_predictions[:, i])),
            'MAE': mean_absolute_error(test_targets[:, i], test_predictions[:, i]),
            'R2': r2_score(test_targets[:, i], test_predictions[:, i])
        },
        'Actual Scale': {
            'MSE': mean_squared_error(test_targets_actual[:, i], test_predictions_actual[:, i]),
            'RMSE': np.sqrt(mean_squared_error(test_targets_actual[:, i], test_predictions_actual[:, i])),
            'MAE': mean_absolute_error(test_targets_actual[:, i], test_predictions_actual[:, i]),
            'R2': r2_score(test_targets_actual[:, i], test_predictions_actual[:, i])
        }
    }

# Print metrics
print("\nTest Set Results:")
print("-" * 50)
for target, scores in metrics.items():
    print(f"\n{target.capitalize()}:")
    print("\nLog Scale Metrics:")
    for metric, value in scores['Log Scale'].items():
        print(f"{metric}: {value:.4f}")
    print("\nActual Scale Metrics:")
    for metric, value in scores['Actual Scale'].items():
        print(f"{metric}: {value:.4f}")

# Create visualizations
print("\n📊 Creating visualizations...")
plt.figure(figsize=(20, 10))

# Plot 1: Log Scale Actual vs Predicted
plt.subplot(2, 3, 1)
plt.scatter(test_targets[:, 0], test_predictions[:, 0], alpha=0.5)
plt.plot([test_targets[:, 0].min(), test_targets[:, 0].max()], 
         [test_targets[:, 0].min(), test_targets[:, 0].max()], 'r--')
plt.xlabel('Actual Likes (log)')
plt.ylabel('Predicted Likes (log)')
plt.title('Likes: Log Scale Actual vs Predicted')

# Plot 2: Actual Scale Actual vs Predicted
plt.subplot(2, 3, 2)
plt.scatter(test_targets_actual[:, 0], test_predictions_actual[:, 0], alpha=0.5)
plt.plot([test_targets_actual[:, 0].min(), test_targets_actual[:, 0].max()], 
         [test_targets_actual[:, 0].min(), test_targets_actual[:, 0].max()], 'r--')
plt.xlabel('Actual Likes')
plt.ylabel('Predicted Likes')
plt.title('Likes: Actual Scale')

# Plot 3: Error Distribution
plt.subplot(2, 3, 3)
errors = test_predictions[:, 0] - test_targets[:, 0]
sns.histplot(errors, kde=True)
plt.xlabel('Prediction Error (log scale)')
plt.title('Likes: Error Distribution')

# Plot 4: Retweets Log Scale
plt.subplot(2, 3, 4)
plt.scatter(test_targets[:, 1], test_predictions[:, 1], alpha=0.5)
plt.plot([test_targets[:, 1].min(), test_targets[:, 1].max()], 
         [test_targets[:, 1].min(), test_targets[:, 1].max()], 'r--')
plt.xlabel('Actual Retweets (log)')
plt.ylabel('Predicted Retweets (log)')
plt.title('Retweets: Log Scale')

# Plot 5: Retweets Actual Scale
plt.subplot(2, 3, 5)
plt.scatter(test_targets_actual[:, 1], test_predictions_actual[:, 1], alpha=0.5)
plt.plot([test_targets_actual[:, 1].min(), test_targets_actual[:, 1].max()], 
         [test_targets_actual[:, 1].min(), test_targets_actual[:, 1].max()], 'r--')
plt.xlabel('Actual Retweets')
plt.ylabel('Predicted Retweets')
plt.title('Retweets: Actual Scale')

# Plot 6: Replies Log Scale
plt.subplot(2, 3, 6)
plt.scatter(test_targets[:, 2], test_predictions[:, 2], alpha=0.5)
plt.plot([test_targets[:, 2].min(), test_targets[:, 2].max()], 
         [test_targets[:, 2].min(), test_targets[:, 2].max()], 'r--')
plt.xlabel('Actual Replies (log)')
plt.ylabel('Predicted Replies (log)')
plt.title('Replies: Log Scale')

plt.tight_layout()
plt.savefig('model_evaluation.png')
print("✅ Saved evaluation plots to 'model_evaluation.png'")

# Save detailed results
results = {
    'metrics': metrics,
    'predictions_log': test_predictions.tolist(),
    'predictions_actual': test_predictions_actual.tolist(),
    'actual_log': test_targets.tolist(),
    'actual_actual': test_targets_actual.tolist()
}

import json
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✅ Saved detailed results to 'evaluation_results.json'")

# Clear memory
del model, base_model, tokenizer, regressor, scaler
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
