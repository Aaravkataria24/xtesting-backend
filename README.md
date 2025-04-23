# TweetLab Backend

This is the backend service for the TweetLab application, deployed on Google Cloud.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python main.py
```

## Deployment

The backend is deployed on Google Cloud. To deploy:

1. Make sure you have the Google Cloud SDK installed
2. Run the deployment command:
```bash
gcloud app deploy
```

## API Endpoints

The backend provides the following API endpoints:

- `/api/tweets` - Get tweet data
- `/api/analyze` - Analyze tweet sentiment
- `/api/stats` - Get tweet statistics

## Environment Variables

Create a `.env` file with the following variables:
```
API_KEY=your_api_key
MODEL_PATH=path_to_model
```

## Model Files

The following model files are required:
- `finetuned_twitter_roberta_multi.pt`
- `regressor_model.pkl`
- `replies_model.pkl`
- `retweets_model.pkl`
- `likes_model.pkl`
- `tfidf_vectorizer.pkl` 