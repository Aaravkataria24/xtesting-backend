import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
from tqdm import tqdm
import time
import os

# Suppress TensorFlow logging in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only shows errors
warnings.filterwarnings('ignore')

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
CHECKPOINT_PATH = "best_model.pt" # previously created fine-tuned roberta model
REGRESSOR_PATH = "regressor_model.pkl" # final regressor model
SCALER_PATH = "scaler.pkl"
CSV_PATH = "processed_tweets_multi.csv"

# Check for GPU and clear memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()  # Clear GPU memory
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 16

# Load dataset
print("📊 Loading dataset...")
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Dataset loaded: {len(train_df)} training samples, {len(val_df)} validation samples")

# Load tokenizer and model
print("🤖 Loading RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

# Custom wrapper for regression head
import torch.nn as nn
class RobertaRegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = base_model
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 3) # reduce size to 3 models

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.regressor(x)

# Move model to device
model = RobertaRegressionHead().to(device)

# Load model checkpoint
print("📥 Loading model checkpoint...")
try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Enhanced embedding generator with feature engineering
def get_enhanced_embeddings(texts, batch_size=BATCH_SIZE, desc="Generating embeddings"):
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=desc):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
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
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return np.concatenate(embeddings, axis=0)

# Generate enhanced embeddings
print("🔍 Generating enhanced embeddings...")
train_embeddings = get_enhanced_embeddings(train_df["content"], desc="Training set embeddings")
val_embeddings = get_enhanced_embeddings(val_df["content"], desc="Validation set embeddings")

# Prepare targets
target_columns = ["likes_log", "retweets_log", "replies_log"]
train_targets = train_df[target_columns].values
val_targets = val_df[target_columns].values

# Normalize targets
target_scaler = StandardScaler()
train_targets_scaled = target_scaler.fit_transform(train_targets)
val_targets_scaled = target_scaler.transform(val_targets)

# Scale features
print("📏 Scaling features...")
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
val_embeddings_scaled = scaler.transform(val_embeddings)

# Save scalers
joblib.dump(scaler, SCALER_PATH)
joblib.dump(target_scaler, "target_scaler.pkl")
print(f"✅ Saved scalers to '{SCALER_PATH}' and 'target_scaler.pkl'")

# Define base models with improved parameters
base_models = [
    ('gb', GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,  # Reduced learning rate
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8, # uses only 80% of the data for each tree to prevent overfitting
        random_state=42
    )),
    ('rf', RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ))
]

# Train separate models for each target
print("🧠 Training models for each target...")
models = {}
val_predictions = np.zeros_like(val_targets_scaled)

for i, target in enumerate(tqdm(target_columns, desc="Training models")):
    print(f"\nTraining model for {target}...")
    start_time = time.time()
    
    # Create stacking regressor with improved parameters
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.03,
            max_depth=4,
            random_state=42
        ),
        cv=3,
        verbose=1
    )
    
    # Train the model with progress tracking
    print(f"Starting training for {target}...")
    print("This may take several minutes...")
    
    # Add timeout check
    max_training_time = 18000  # 300 minutes (I think)
    training_start = time.time()
    
    try:
        stacking_regressor.fit(train_embeddings_scaled, train_targets_scaled[:, i])
        training_time = time.time() - start_time
        print(f"Training completed for {target} in {training_time:.2f} seconds")
        
        # Make predictions
        print(f"Making predictions for {target}...")
        val_predictions[:, i] = stacking_regressor.predict(val_embeddings_scaled)
        
        # Store the model
        models[target] = stacking_regressor
        
        # Print feature importance
        print(f"\nFeature Importance for {target}:")
        for name, model in base_models:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features = np.argsort(importances)[-5:][::-1]
                print(f"{name} top 5 important features: {top_features}")
        
        # Save intermediate results
        print(f"Saving intermediate results for {target}...")
        joblib.dump(models, f"regressor_model_{target}.pkl")
        print(f"✅ Saved intermediate model for {target}")
        
    except Exception as e:
        print(f"Error during training of {target}: {str(e)}")
        raise
    
    # Check if training is taking too long
    if time.time() - training_start > max_training_time:
        print(f"Warning: Training for {target} is taking longer than {max_training_time/60} minutes")
        print("Consider interrupting and reducing model complexity further")

# Denormalize predictions for final metrics
val_predictions_denorm = target_scaler.inverse_transform(val_predictions)

# Calculate overall metrics
val_mse = mean_squared_error(val_targets, val_predictions_denorm)
val_r2 = r2_score(val_targets, val_predictions_denorm)

print("\nOverall Validation Results:")
print(f"Mean Squared Error: {val_mse:.4f}")
print(f"R² Score: {val_r2:.4f}")

# Print per-target metrics
print("\nPer-target Validation Results:")
for i, target in enumerate(target_columns):
    target_mse = mean_squared_error(val_targets[:, i], val_predictions_denorm[:, i])
    target_r2 = r2_score(val_targets[:, i], val_predictions_denorm[:, i])
    print(f"\n{target}:")
    print(f"Mean Squared Error: {target_mse:.4f}")
    print(f"R² Score: {target_r2:.4f}")

# Save the trained models
joblib.dump(models, REGRESSOR_PATH)
print(f"\n✅ Saved regressor models to '{REGRESSOR_PATH}'")
