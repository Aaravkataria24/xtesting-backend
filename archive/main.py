from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import joblib
import uvicorn
import os
import gc

# Constants
MODEL_PATH = "finetuned_twitter_roberta_multi.pt"
REGRESSOR_PATH = "regressor_model.pkl"
SCALER_PATH = "scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"
BASE_MODEL = "cardiffnlp/twitter-roberta-base"

# Custom scaler class
class MultiTargetScaler:
    def __init__(self):
        # These means and stds are for log1p-transformed values
        # Manually hardcoded to avoid loading the scaler from the training data
        self.means = np.array([2.0, 1.0, 0.5])  # Means for log1p(likes), log1p(retweets), log1p(replies)
        self.stds = np.array([1.5, 1.0, 0.8])   # Standard deviations for log-transformed values
        
    def transform(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return (X - self.means) / self.stds
    
    def inverse_transform(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return (X * self.stds) + self.means

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU memory if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Load tokenizer and model
print("Loading models and scalers...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    roberta = AutoModel.from_pretrained(BASE_MODEL)
except Exception as e:
    raise RuntimeError(f"Failed to load RoBERTa model: {str(e)}")

# Custom wrapper for regression head - essentially just a linear layer
class RobertaRegressionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = roberta
        self.dropout = torch.nn.Dropout(0.2)
        self.regressor = torch.nn.Linear(self.roberta.config.hidden_size, 3)
        
        # Add position_ids to match saved model
        position_ids = torch.arange(512).expand((1, -1))
        self.roberta.embeddings.register_buffer('position_ids', position_ids)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.regressor(x)

model = RobertaRegressionHead().to(device)
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Successfully loaded model checkpoint")
except Exception as e:
    raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
model.eval()

# Load regressor and scalers
try:
    regressor = joblib.load(REGRESSOR_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load regressor or scalers: {str(e)}")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://xtester.netlify.app",
        "https://xtesting.aaravkataria.com",
        # Add your Google Cloud frontend URL when available
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schemas
class TweetInput(BaseModel):
    text: str

class SplitTestInput(BaseModel):
    tweet1: str
    tweet2: str

# Enhanced embedding generation
@torch.no_grad()
def get_enhanced_embeddings(text):
    encodings = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
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
    
    return combined_features.cpu().numpy()

# Helper function to get rating
def get_rating(score):
    if score < 5:
        return "very bad"
    elif score < 20:
        return "bad"
    elif score < 50:
        return "decent"
    elif score < 100:
        return "good"
    else:
        return "very good"

# Embedding + Prediction
@torch.no_grad()
def get_prediction(text):
    print(f"\n🔍 Predicting for tweet: {text}")
    try:
        # Generate enhanced embeddings
        embeddings = get_enhanced_embeddings(text)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Scale the embeddings
        embeddings_scaled = scaler.transform(embeddings)
        print(f"Scaled embeddings shape: {embeddings_scaled.shape}")
        
        # Get predictions for each target
        predictions = np.zeros((1, 3))
        targets = ['likes_log', 'retweets_log', 'replies_log']
        
        for i, target in enumerate(targets):
            pred = regressor[target].predict(embeddings_scaled)
            predictions[0, i] = pred
            print(f"Raw prediction for {target}: {pred}")
        
        print(f"All raw predictions: {predictions}")
        
        # Denormalize predictions
        predictions_denorm = target_scaler.inverse_transform(predictions)
        print(f"Predictions after denormalization: {predictions_denorm}")
        
        # Convert log predictions to actual numbers using expm1 (inverse of log1p)
        predictions_actual = np.expm1(predictions_denorm).astype(int)
        print(f"Predictions after expm1: {predictions_actual}")
        
        # Apply reasonable caps to predictions
        caps = np.array([5000, 500, 200])  # Maximum values for likes, retweets, replies
        predictions_actual = np.minimum(predictions_actual, caps)
        
        # Ensure predictions are non-negative
        predictions_actual = np.maximum(predictions_actual, 0)
        
        # Scale down likes by dividing by 100 (more aggressive scaling)
        predictions_actual[0][0] = predictions_actual[0][0] // 100
        
        # Scale down retweets and replies
        predictions_actual[0][1] = predictions_actual[0][1] // 10
        predictions_actual[0][2] = predictions_actual[0][2] // 5
        
        # Calculate engagement score
        engagement_score = int(np.sum(predictions_actual))
        
        # Get rating
        rating = get_rating(engagement_score)
        
        result = {
            "likes": int(predictions_actual[0][0]),
            "retweets": int(predictions_actual[0][1]),
            "replies": int(predictions_actual[0][2]),
            "engagement_score": engagement_score,
            "rating": rating
        }
        print(f"✅ Final prediction: {result}")
        return result
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Health check endpoint for Cloud Run
@app.get("/health")
def health_check():
    return {"status": "ok"}

# API routes
@app.post("/predict/single")
def predict_single(input: TweetInput):
    try:
        return {"prediction": get_prediction(input.text)}
    except Exception as e:
        print(f"❌ Error in predict_single: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/split")
def predict_split(input: SplitTestInput):
    try:
        pred1 = get_prediction(input.tweet1)
        pred2 = get_prediction(input.tweet2)
        winner = "tweet1" if pred1["engagement_score"] > pred2["engagement_score"] else "tweet2"
        return {
            "tweet1": pred1,
            "tweet2": pred2,
            "better_tweet": winner
        }
    except Exception as e:
        print(f"❌ Error in predict_split: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use 8080 instead of 8000 for Google Cloud
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")