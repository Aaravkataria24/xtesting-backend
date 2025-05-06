import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Create a custom scaler class
class MultiTargetScaler:
    def __init__(self):
        self.means = np.array([10.0, 2.0, 1.0])  # Approximate means for likes, retweets, replies
        self.stds = np.array([20.0, 5.0, 3.0])   # Approximate standard deviations
        
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

try:
    # Create and test the scaler
    print("Creating scaler...")
    target_scaler = MultiTargetScaler()
    
    # Test with sample data
    test_input = np.array([[5, 1, 0]])
    print(f"Test input: {test_input}")
    
    transformed = target_scaler.transform(test_input)
    print(f"Transformed: {transformed}")
    
    inverse_transformed = target_scaler.inverse_transform(transformed)
    print(f"Inverse transformed: {inverse_transformed}")
    
    # Save the scaler
    joblib.dump(target_scaler, 'target_scaler.pkl')
    print("✅ Successfully created and saved target_scaler.pkl")

except Exception as e:
    print(f"❌ Error: {str(e)}") 