import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import json
import sys
from torch import nn
import torch.nn.functional as F
from datetime import datetime

# Define the model architecture (same as in fine_tune_roberta.py)
class EnhancedRobertaRegressor(nn.Module):
    def __init__(self, num_features=0, dropout_rate=0.2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        hidden_size = self.roberta.config.hidden_size  # 768 for RoBERTa base
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

class TweetPredictor:
    def __init__(self, model_path=None, normalization_params_path=None, device=None):
        if model_path is None:
            model_path = "finetuned_twitter_roberta_multi.pt"
        if normalization_params_path is None:
            normalization_params_path = "normalization_params.json"
        # Load normalization parameters
        with open(normalization_params_path, 'r') as f:
            self.norm_params = json.load(f)
        print(f"Loaded normalization parameters: {self.norm_params.keys()}")
        # Set feature columns to the new 9-feature list
        self.feature_columns = [
            'follower_count_log', 'length_log', 'has_image', 'has_video',
            'has_link', 'has_mention', 'has_crypto_mention', 'hour', 'minute'
        ]
        self.has_features = True
        self.num_features = 9
        print(f"Feature columns: {self.feature_columns}")
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
        self.model = EnhancedRobertaRegressor(num_features=self.num_features).to(self.device)
        # Load model weights (weights-only file)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("Model loaded successfully!")
        
    def normalize_features(self, features_df):
        feature_means = [self.norm_params['feature_means'][self.norm_params['feature_columns'].index(col)] for col in self.feature_columns]
        feature_stds = [self.norm_params['feature_stds'][self.norm_params['feature_columns'].index(col)] for col in self.feature_columns]
        feature_stds = [std if std != 0 else 1.0 for std in feature_stds]
        # Handle time_posted if present - convert to hour and minute
        if 'time_posted' in features_df.columns and 'hour' not in features_df.columns:
            try:
                features_df['hour'], features_df['minute'] = zip(*features_df['time_posted'].apply(self._extract_time))
                print(f"Converted time_posted to hour and minute features.")
            except Exception as e:
                print(f"Error converting time_posted: {e}. Using default values.")
                features_df['hour'] = 12
                features_df['minute'] = 0
        features = []
        for i, col in enumerate(self.feature_columns):
            if col in features_df.columns:
                normalized_feature = (features_df[col].values - feature_means[i]) / feature_stds[i]
                features.append(normalized_feature)
            else:
                print(f"Warning: Feature '{col}' not found in input data. Using zeros.")
                features.append(np.zeros(len(features_df)))
        return np.column_stack(features)
    
    def _extract_time(self, time_str):
        """Extract hour and minute from time string in various formats"""
        try:
            # Try parsing ISO 8601 format (e.g., "2025-06-11T06:44:56.016Z")
            if 'T' in str(time_str):
                dt = datetime.fromisoformat(str(time_str).replace('Z', '+00:00'))
                return dt.hour, dt.minute
            
            # Try direct parsing of HH:MM:SS or HH:MM format
            parts = str(time_str).split(':')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
            
            # Try parsing as datetime if it's a different format
            dt = datetime.strptime(str(time_str), "%H:%M:%S")
            return dt.hour, dt.minute
        except Exception as e:
            # Return default values if parsing fails
            print(f"Could not parse time: {time_str}. Using default values. Error: {str(e)}")
            return 12, 0  # Default to noon
    
    def denormalize_predictions(self, predictions, features_df):
        """Convert normalized predictions back to original scale, using follower_count_log"""
        target_means = self.norm_params['target_means']
        target_stds = self.norm_params['target_stds']
        target_columns = self.norm_params['target_columns']

        # Get follower_count_log from features_df
        follower_count_log = features_df['follower_count_log'].values.reshape(-1, 1)  # shape (batch, 1)

        denormalized = {}
        for i, col in enumerate(target_columns):
            # Undo normalization
            pred_log_norm = predictions[:, i] * target_stds[i] + target_means[i]
            # Multiply by follower_count_log to get back to log scale
            pred_log = pred_log_norm * follower_count_log[:, 0]
            # De-log
            denormalized[col.replace('_log_norm', '')] = np.exp(pred_log) - 1

        return denormalized
    
    def predict(self, texts, features_df=None):
        """Make predictions for a list of texts with optional features"""
        # Tokenize texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Process features if available
        features_tensor = None
        if features_df is not None and self.has_features:
            normalized_features = self.normalize_features(features_df)
            if normalized_features is not None:
                features_tensor = torch.tensor(normalized_features, dtype=torch.float32).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, features=features_tensor)
            predictions = outputs.cpu().numpy()
        
        # Denormalize predictions
        results = self.denormalize_predictions(predictions, features_df)
        
        # Add texts to results
        results['text'] = texts
        
        return pd.DataFrame(results)

def display_available_features():
    """Display the list of available features that can be used with the model"""
    try:
        with open("normalization_params.json", 'r') as f:
            params = json.load(f)
        
        if 'feature_columns' in params:
            print("\n🔍 Available Features:")
            for i, feature in enumerate(params['feature_columns']):
                print(f"  {i+1}. {feature}")
            
            # Check if time features are present in a special way
            if 'hour' in params['feature_columns'] and 'minute' in params['feature_columns']:
                print("\n⏰ Time Features:")
                print("  You can provide 'time_posted' in HH:MM:SS format instead of separate hour and minute.")
                print("  Example: time_posted=14:30:00")
            
            print("\nUse these feature names when providing feature values.")
        else:
            print("❌ No feature information found in normalization parameters.")
    except FileNotFoundError:
        print("❌ Normalization parameters file not found.")
    except json.JSONDecodeError:
        print("❌ Error reading normalization parameters file.")

def main():
    # Check if we just want to display available features
    if len(sys.argv) > 1 and sys.argv[1] == "--features":
        display_available_features()
        sys.exit(0)
        
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python predict.py <text_to_predict> [<feature_values>]")
        print("Example: python predict.py \"This is an amazing tweet about AI!\" \"has_image=1,has_video=0,time_posted=14:30:00\"")
        print("Use --features flag to see available features: python predict.py --features")
        sys.exit(1)
    
    # Get text and optional features
    text = sys.argv[1]
    features_dict = {}
    
    if len(sys.argv) > 2:
        feature_str = sys.argv[2]
        # Parse feature string like "has_image=1,has_video=0"
        feature_pairs = feature_str.split(',')
        for pair in feature_pairs:
            if '=' in pair:
                key, value = pair.split('=')
                try:
                    # Convert numeric values
                    features_dict[key] = float(value)
                except ValueError:
                    # Keep string values as is
                    features_dict[key] = value
    
    # Load predictor
    try:
        model_path = "checkpoints/best_model.pt"  # Update path as needed
        norm_params_path = "normalization_params.json"
        predictor = TweetPredictor(model_path, norm_params_path)
        
        # Make prediction
        features_df = pd.DataFrame([features_dict]) if features_dict else None
        results = predictor.predict([text], features_df)
        
        # Print results
        print("\n🔮 Prediction Results 🔮")
        print(f"Text: {text}")
        print(f"Predicted Likes: {int(results['likes'].values[0])}")
        print(f"Predicted Retweets: {int(results['retweets'].values[0])}")
        print(f"Predicted Replies: {int(results['replies'].values[0])}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 