import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import argparse
from predict import TweetPredictor
import matplotlib.pyplot as plt
from datetime import datetime

def load_test_data(test_file):
    """Load and preprocess test data"""
    df = pd.read_csv(test_file)
    
    # Convert numeric columns to float
    numeric_cols = ['follower_count', 'view_count', 'length', 'likes', 'retweets', 'replies']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert boolean columns to int
    bool_cols = ['has_image', 'has_video', 'has_link', 'has_mention', 'has_crypto_mention', 'is_quoting', 'has_poll']
    for col in bool_cols:
        # Convert various boolean representations to int
        df[col] = df[col].map(lambda x: 1 if str(x).lower() in ['true', 'yes', '1', 't', 'y'] else 0)
    
    # Convert engagement metrics to log scale
    for col in ['likes', 'retweets', 'replies']:
        df[f'{col}_log'] = np.log(df[col] + 1)
    
    # Convert numeric columns to log scale
    df['follower_count_log'] = np.log(df['follower_count'] + 1)
    df['view_count_log'] = np.log(df['view_count'] + 1)
    df['length_log'] = np.log(df['length'] + 1)
    
    # Extract hour and minute from time_posted
    df[['hour', 'minute']] = df['time_posted'].str.split(':', expand=True).iloc[:, [0, 1]].astype(int)
    
    # Rename content column to text if needed
    if 'content' in df.columns and 'text' not in df.columns:
        df['text'] = df['content']
    
    return df

def calculate_metrics(y_true, y_pred, target_name):
    """Calculate various regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage within 5% of true value
    within_5_percent = np.mean(np.abs(y_pred - y_true) / y_true <= 0.05) * 100
    
    # Calculate percentage within different error margins
    within_10_percent = np.mean(np.abs(y_pred - y_true) / y_true <= 0.10) * 100
    within_20_percent = np.mean(np.abs(y_pred - y_true) / y_true <= 0.20) * 100
    
    return {
        'target': target_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Within_5%': within_5_percent,
        'Within_10%': within_10_percent,
        'Within_20%': within_20_percent
    }

def plot_predictions(y_true, y_pred, target_name, output_dir):
    """Create scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Perfect prediction line
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual {target_name}')
    
    # Add R² value to plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    
    plt.savefig(f'{output_dir}/{target_name.lower()}_predictions.png')
    plt.close()

def evaluate_model(test_file, model_path, norm_params_path, output_dir, batch_size=32):
    """Evaluate model performance on test data"""
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(test_file)
    
    # Initialize predictor
    print("Loading model...")
    predictor = TweetPredictor(model_path, norm_params_path)
    
    # Prepare features DataFrame
    feature_columns = predictor.norm_params['feature_columns']
    features_df = test_data[feature_columns].copy()
    
    # Make predictions in batches
    print("Making predictions...")
    all_predictions = []
    for i in range(0, len(test_data), batch_size):
        batch_texts = test_data['text'].iloc[i:i+batch_size].tolist()
        batch_features = features_df.iloc[i:i+batch_size]
        
        batch_predictions = predictor.predict(batch_texts, batch_features)
        all_predictions.append(batch_predictions)
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calculate metrics for each target
    print("\nCalculating metrics...")
    metrics = []
    for target in ['likes', 'retweets', 'replies']:
        target_metrics = calculate_metrics(
            test_data[target].values,
            predictions_df[target].values,
            target
        )
        metrics.append(target_metrics)
        
        # Create prediction plots
        plot_predictions(
            test_data[target].values,
            predictions_df[target].values,
            target,
            output_dir
        )
    
    # Save metrics to file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{output_dir}/evaluation_metrics.csv', index=False)
    
    # Print summary
    print("\n📊 Evaluation Results 📊")
    print(metrics_df.to_string(index=False))
    
    # Save example predictions
    print("\nSaving example predictions...")
    example_df = pd.DataFrame({
        'text': test_data['text'],
        'actual_likes': test_data['likes'],
        'predicted_likes': predictions_df['likes'],
        'actual_retweets': test_data['retweets'],
        'predicted_retweets': predictions_df['retweets'],
        'actual_replies': test_data['replies'],
        'predicted_replies': predictions_df['replies']
    })
    example_df.to_csv(f'{output_dir}/example_predictions.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate tweet engagement prediction model')
    parser.add_argument('test_file', help='Path to test data CSV file')
    parser.add_argument('--model_path', default='checkpoints/best_model.pt', help='Path to trained model')
    parser.add_argument('--norm_params_path', default='normalization_params.json', help='Path to normalization parameters')
    parser.add_argument('--output_dir', default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for predictions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_model(
        args.test_file,
        args.model_path,
        args.norm_params_path,
        args.output_dir,
        args.batch_size
    )

if __name__ == "__main__":
    main() 