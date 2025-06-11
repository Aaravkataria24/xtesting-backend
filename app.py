from fastapi import FastAPI, HTTPException, Request, Response, Cookie, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from predict import TweetPredictor
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests
import base64
import hashlib
import secrets
from fastapi.responses import JSONResponse, RedirectResponse

# Load environment variables from .env file
load_dotenv(override=True)  # override=True will override any existing env vars

# Debug print environment variables
print("\n=== Environment Variables ===")
print(f"X_CLIENT_ID: {os.getenv('X_CLIENT_ID')}")
print(f"X_REDIRECT_URI: {os.getenv('X_REDIRECT_URI')}")
print("===========================\n")

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "http://localhost:3000",  # Alternative local port
        "https://xtester.netlify.app",  # Old frontend
        "https://xtesting.aaravkataria.com",  # Old frontend
        "http://localhost:5174",  # New frontend local development
        "https://tweetlab.aaravkataria.com"  # New frontend production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = TweetPredictor(
    model_path="checkpoints/best_model.pt",
    normalization_params_path="normalization_params.json"
)

# X API credentials
X_CLIENT_ID = os.getenv('X_CLIENT_ID')
X_CLIENT_SECRET = os.getenv('X_CLIENT_SECRET')
X_REDIRECT_URI = os.getenv('X_REDIRECT_URI')

class TweetRequest(BaseModel):
    text: str
    has_image: bool = False
    has_video: bool = False
    has_link: bool = False
    has_mention: bool = False
    has_crypto_mention: bool = False
    is_quoting: bool = False
    has_poll: bool = False
    time_posted: str
    follower_count: int
    view_count: Optional[int] = None
    length: Optional[int] = None

@app.post("/predict")
async def predict_engagement(request: TweetRequest) -> Dict[str, Any]:
    try:
        # Create features dictionary with log transformations
        features = {
            "has_image": int(request.has_image),
            "has_video": int(request.has_video),
            "has_link": int(request.has_link),
            "has_mention": int(request.has_mention),
            "has_crypto_mention": int(request.has_crypto_mention),
            "is_quoting": int(request.is_quoting),
            "has_poll": int(request.has_poll),
            "time_posted": request.time_posted,
            "follower_count": request.follower_count,
            "view_count": request.view_count if request.view_count is not None else 0,
            "length": request.length if request.length is not None else len(request.text),
            # Add log-transformed features
            "follower_count_log": np.log(request.follower_count + 1),
            "view_count_log": np.log((request.view_count if request.view_count is not None else 0) + 1),
            "length_log": np.log((request.length if request.length is not None else len(request.text)) + 1)
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Get predictions
        predictions = predictor.predict([request.text], features_df)
        
        return {
            "likes": int(predictions["likes"][0]),
            "retweets": int(predictions["retweets"][0]),
            "replies": int(predictions["replies"][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/x/callback")
async def x_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    code_verifier: Optional[str] = Query(None)
):
    print("=== Callback Debug Info ===")
    print(f"Query params: {request.query_params}")
    print(f"Headers: {request.headers}")
    print(f"Cookies: {request.cookies}")
    
    if not code or not state:
        print("Missing code or state in query parameters")
        raise HTTPException(status_code=400, detail="Missing code or state parameters")
    
    # PKCE and state are handled by the frontend; just use the provided code_verifier
    if not code_verifier:
        print("Missing code_verifier in query params")
        raise HTTPException(status_code=400, detail="Missing code_verifier parameter")
    # Do NOT override code_verifier here; use the one from the query param

    try:
        print("Attempting to exchange code for token...")
        # Create Basic Auth header
        auth_string = f"{X_CLIENT_ID}:{X_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('ascii')
        base64_auth = base64.b64encode(auth_bytes).decode('ascii')
        
        # Debug print the auth header (without the actual secret)
        print(f"Using client ID: {X_CLIENT_ID}")
        print(f"Using redirect URI: {X_REDIRECT_URI}")
        print(f"Auth header format: Basic {base64_auth[:10]}...")
        
        # Exchange code for access token
        token_data = {
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': X_REDIRECT_URI,
            'code_verifier': code_verifier
        }
        
        print("Token request data:", {k: v for k, v in token_data.items() if k != 'code'})
        
        token_response = requests.post(
            'https://api.twitter.com/2/oauth2/token',
            data=token_data,
            headers={
                'Authorization': f'Basic {base64_auth}',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
        )
        
        print(f"Token response status: {token_response.status_code}")
        print(f"Token response: {token_response.text}")
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        access_token = token_response.json()['access_token']
        
        print("Attempting to get user data...")
        # Get user data
        user_response = requests.get(
            'https://api.twitter.com/2/users/me',
            headers={'Authorization': f'Bearer {access_token}'},
            params={'user.fields': 'public_metrics,profile_image_url'}
        )
        
        print(f"User response status: {user_response.status_code}")
        print(f"User response: {user_response.text}")
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user data")
        
        user_data = user_response.json()['data']
        # Return user info as JSON
        return {
            "username": user_data['username'],
            "followerCount": user_data['public_metrics']['followers_count'],
            "profileImageUrl": user_data['profile_image_url']
        }
        
    except Exception as e:
        print(f"Error in callback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/x/login")
async def x_login(response: Response):
    # Generate state and code_verifier
    state = secrets.token_urlsafe(16)
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b'=') .decode('ascii')

    # Set cookies for state and code_verifier
    response.set_cookie("x_auth_state", state, httponly=True, samesite="lax")
    response.set_cookie("code_verifier", code_verifier, httponly=True, samesite="lax")

    # Build the X OAuth2 URL
    params = {
        "response_type": "code",
        "client_id": X_CLIENT_ID,
        "redirect_uri": X_REDIRECT_URI,
        "scope": "tweet.read users.read offline.access",  # adjust scopes as needed
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256"
    }
    from urllib.parse import urlencode
    url = f"https://twitter.com/i/oauth2/authorize?{urlencode(params)}"
    return RedirectResponse(url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 