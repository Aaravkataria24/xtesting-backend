import torch
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch.nn.functional as F
import pickle

# Load normalization parameters
with open('normalization_params.json', 'r') as f:
    norm_params = json.load(f)
target_means = np.array(norm_params['means'])
target_stds = np.array(norm_params['stds'])

print("Normalization Parameters:")
print("-" * 50)
print(f"Means: {target_means}")
print(f"Standard Deviations: {target_stds}")

# Load the model checkpoint on CPU with error handling
checkpoint_path = "best_model.pt"
try:
    # First try loading with pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
except Exception as e1:
    try:
        # If that fails, try torch.load
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e2:
        print(f"Error loading checkpoint: {str(e2)}")
        raise

# Print basic information
print("\nModel Checkpoint Analysis:")
print("-" * 50)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['loss']:.4f}")
print(f"Within 5% error: {checkpoint['within_5_percent']:.2f}%")

# Analyze model state dict
model_state = checkpoint['model_state_dict']
print("\nModel Architecture:")
print("-" * 50)
for key, value in model_state.items():
    if 'weight' in key or 'bias' in key:
        print(f"{key}: Shape {value.shape}")

# Analyze optimizer state
optimizer_state = checkpoint['optimizer_state_dict']
print("\nOptimizer State:")
print("-" * 50)
print(f"Learning rate: {optimizer_state['param_groups'][0]['lr']}")
print(f"Weight decay: {optimizer_state['param_groups'][0]['weight_decay']}")

# Print model size
total_params = sum(p.numel() for p in model_state.values())
print(f"\nTotal parameters: {total_params:,}")

# Calculate prediction ranges
print("\nPrediction Accuracy Analysis:")
print("-" * 50)

# For a prediction of 100 likes
predicted_likes = 100

# Convert prediction to normalized space
normalized_pred = (predicted_likes - target_means[0]) / target_stds[0]

# Calculate error ranges based on validation loss
val_loss = checkpoint['loss']
std_error = np.sqrt(val_loss)  # Approximate standard error

# Calculate ranges for different confidence levels
ranges = {
    '68% confidence': 1 * std_error,
    '95% confidence': 2 * std_error,
    '99% confidence': 3 * std_error
}

print(f"\nFor a prediction of {predicted_likes} likes:")
for confidence, error in ranges.items():
    # Convert error back to actual space
    actual_error = error * target_stds[0]
    lower_bound = max(0, predicted_likes - actual_error)
    upper_bound = predicted_likes + actual_error
    print(f"{confidence}: Actual likes likely between {lower_bound:.0f} and {upper_bound:.0f}")

# Calculate percentage error distribution
print("\nError Distribution:")
print("-" * 50)
print(f"Within 5% error: {checkpoint['within_5_percent']:.2f}% of predictions")
print(f"Standard Error: {std_error:.4f} (in normalized space)")
print(f"Average Error: {std_error * target_stds[0]:.2f} likes")

# Save analysis to file
analysis = {
    'epoch': checkpoint['epoch'],
    'val_loss': checkpoint['loss'],
    'within_5_percent': checkpoint['within_5_percent'],
    'total_parameters': total_params,
    'model_architecture': {key: str(value.shape) for key, value in model_state.items() if 'weight' in key or 'bias' in key},
    'optimizer_settings': {
        'learning_rate': optimizer_state['param_groups'][0]['lr'],
        'weight_decay': optimizer_state['param_groups'][0]['weight_decay']
    },
    'normalization_parameters': {
        'means': target_means.tolist(),
        'stds': target_stds.tolist()
    },
    'prediction_accuracy': {
        'standard_error': float(std_error),
        'average_error_likes': float(std_error * target_stds[0]),
        'confidence_ranges': {
            confidence: {
                'lower': float(max(0, predicted_likes - error * target_stds[0])),
                'upper': float(predicted_likes + error * target_stds[0])
            }
            for confidence, error in ranges.items()
        }
    }
}

with open('model_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\nAnalysis saved to model_analysis.json") 