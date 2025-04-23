from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import joblib
import uvicorn
import os

# Constants
MODEL_PATH = "finetuned_twitter_roberta_multi.pt"
REGRESSOR_PATH = "regressor_model.pkl"
BASE_MODEL = "cardiffnlp/twitter-roberta-base"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
roberta = AutoModel.from_pretrained(BASE_MODEL)

class RobertaRegressionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = roberta
        self.dropout = torch.nn.Dropout(0.2)
        self.regressor = torch.nn.Linear(self.roberta.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.regressor(x)

model = RobertaRegressionHead().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load regressor
regressor = joblib.load(REGRESSOR_PATH)

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

# Embedding + Prediction
@torch.no_grad()
def get_prediction(text):
    print(f"\n🔍 Predicting for tweet: {text}")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    prediction = outputs.cpu().numpy()
    prediction_exp = np.expm1(prediction).astype(int)

    result = {
        "likes": int(prediction_exp[0][0]),
        "retweets": int(prediction_exp[0][1]),
        "replies": int(prediction_exp[0][2]),
        "engagement_score": int(np.sum(prediction_exp))
    }
    print(f"✅ Prediction: {result}")
    return result

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