# Using Your Trained RoBERTa Model

Now that you've successfully trained your RoBERTa model in Google Colab, here's how to use it with your existing `predict.py` script.

## 1. Download Files from Google Colab

Download these files from Google Colab to your local machine:
- The best model: `/content/drive/MyDrive/tweetlab_model/checkpoints/best_model.pt`
- Normalization parameters: `/content/normalization_params.json`

## 2. Set Up Your Files

Place the files in these locations in your project directory:
```
/Users/aarav/Downloads/tweetlab-backend/checkpoints/best_model.pt
/Users/aarav/Downloads/tweetlab-backend/normalization_params.json
```

Make sure you have a `checkpoints` directory in your project folder. If not, create it:
```bash
mkdir -p checkpoints
```

## 3. Make Predictions Using predict.py

Your existing `predict.py` script is already set up to use the model. You can use it from the command line:

```bash
python predict.py "This is a test tweet about #AI" "has_image=0,has_video=0,has_link=1,follower_count_log=10.5,view_count_log=10.3,length_log=5.5"
```

To see what features are available for the model:

```bash
python predict.py --features
```

## 4. Understanding the Results

The script will output:
- Predicted Likes: Number of expected likes
- Predicted Retweets: Number of expected retweets
- Predicted Replies: Number of expected replies

## 5. Integrating in Your Application

The `TweetPredictor` class in `predict.py` can be imported and used in your application:

```python
from predict import TweetPredictor

# Initialize the predictor
predictor = TweetPredictor(
    model_path="checkpoints/best_model.pt",
    normalization_params_path="normalization_params.json"
)

# Make predictions
tweet_text = "This is a tweet about AI and machine learning #AI"
features = {
    'has_image': 0,
    'has_video': 0,
    'has_link': 1,
    'has_mention': 0,
    'follower_count_log': 10.5,
    'view_count_log': 10.3,
    'length_log': 5.5,
    'hour': 14,   # If the tweet was posted at 2:30 PM
    'minute': 30
}

features_df = pd.DataFrame([features])
results = predictor.predict([tweet_text], features_df)

# Access results
print(f"Predicted Likes: {int(results['likes'].values[0])}")
print(f"Predicted Retweets: {int(results['retweets'].values[0])}")
print(f"Predicted Replies: {int(results['replies'].values[0])}")
```

Remember to have the required dependencies installed:
```bash
pip install torch transformers pandas numpy
``` 