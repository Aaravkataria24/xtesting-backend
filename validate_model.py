import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from predict import TweetPredictor
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

def load_and_preprocess_data(file_path):
    """Load and preprocess the validation data"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Set all follower counts to 350
    df['follower_count'] = 350
    
    # Convert date_posted and time_posted to datetime
    df['datetime'] = pd.to_datetime(df['date_posted'] + ' ' + df['time_posted'])
    
    # Filter out tweets from last 48 hours
    cutoff_time = datetime.now() - timedelta(hours=48)
    df = df[df['datetime'] < cutoff_time]
    
    # Convert boolean columns
    bool_columns = ['has_image', 'has_video', 'has_link', 'has_mention', 'has_crypto_mention', 'has_poll']
    for col in bool_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # Extract hour and minute from time_posted
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Apply log transformation to follower_count and length
    df['follower_count_log'] = np.log1p(df['follower_count'])
    df['length_log'] = np.log1p(df['length'])
    
    print(f"Set all follower counts to 350 (log value: {np.log1p(350):.2f})")
    return df

def calculate_accuracy_buckets(actual, predicted):
    """Calculate percentage of predictions within different accuracy ranges"""
    # Calculate percentage error
    percent_error = np.abs((actual - predicted) / actual) * 100
    
    # Define accuracy buckets
    buckets = {
        'within_5%': 0,
        'within_10%': 0,
        'within_20%': 0,
        'within_50%': 0,
        'over_50%': 0
    }
    
    # Count predictions in each bucket
    total = len(percent_error)
    buckets['within_5%'] = np.sum(percent_error <= 5) / total * 100
    buckets['within_10%'] = np.sum(percent_error <= 10) / total * 100
    buckets['within_20%'] = np.sum(percent_error <= 20) / total * 100
    buckets['within_50%'] = np.sum(percent_error <= 50) / total * 100
    buckets['over_50%'] = np.sum(percent_error > 50) / total * 100
    
    return buckets

def plot_accuracy_distribution(actual, predicted, metric_name):
    """Plot the distribution of prediction accuracy"""
    percent_error = np.abs((actual - predicted) / actual) * 100
    
    plt.figure(figsize=(10, 6))
    plt.hist(percent_error, bins=50, range=(0, 100))
    plt.title(f'Distribution of Prediction Error for {metric_name}')
    plt.xlabel('Percentage Error')
    plt.ylabel('Number of Predictions')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{metric_name}_error_distribution.png')
    plt.close()

def main():
    # Initialize the predictor
    predictor = TweetPredictor(
        model_path="checkpoints/best_model.pt",
        normalization_params_path="normalization_params.json"
    )
    
    # Load and preprocess data
    print("Loading and preprocessing validation data...")
    df = load_and_preprocess_data('temp_4.csv')
    
    # Prepare features for prediction
    feature_columns = [
        'follower_count_log', 'content', 'hour', 'minute',
        'has_image', 'has_video', 'has_link', 'has_mention',
        'has_crypto_mention', 'length_log', 'has_poll'
    ]
    
    features_df = df[feature_columns].copy()
    
    # Make predictions
    print("Making predictions...")
    results = predictor.predict(features_df['content'].tolist(), features_df)
    
    # Calculate accuracy metrics for each target
    metrics = ['likes', 'retweets', 'replies']
    overall_results = {}
    
    for metric in metrics:
        print(f"\nAnalyzing {metric} predictions...")
        
        # Calculate accuracy buckets
        buckets = calculate_accuracy_buckets(df[metric].values, results[metric].values)
        overall_results[metric] = buckets
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(df[metric].values, results[metric].values) * 100
        
        # Print results
        print(f"\n{metric.upper()} Prediction Accuracy:")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print("\nAccuracy Distribution:")
        for bucket, percentage in buckets.items():
            print(f"{bucket}: {percentage:.2f}%")
        
        # Plot error distribution
        plot_accuracy_distribution(df[metric].values, results[metric].values, metric)
    
    # Save detailed results to JSON
    with open('validation_results.json', 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    print("\nValidation complete! Results saved to validation_results.json")
    print("Error distribution plots saved as PNG files.")

if __name__ == "__main__":
    main() 