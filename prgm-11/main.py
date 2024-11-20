'''
pip install numpy
pip install matplotlib
'''

import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Linear Regression function
def locally_weighted_regression(X, y, tau, x_query):
    """
    X: Training data (features)
    y: Target values
    tau: Bandwidth parameter controlling the locality of the regression
    x_query: Query point (for which we are predicting the value)
    """
    m, n = X.shape  # Number of training examples and features
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (column of ones)
    
    # Calculate weights based on Gaussian kernel
    weights = np.exp(-np.sum((X_b - x_query)**2, axis=1) / (2 * tau**2))
    W = np.diag(weights)  # Create diagonal weight matrix
    
    # Perform the weighted least squares regression
    theta = np.linalg.inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
    
    return theta[0] + theta[1:] @ x_query  # Return predicted value

# Generate synthetic data for demonstration
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # Sinusoidal data with noise

# Hyperparameters
tau = 1.0  # Bandwidth parameter (controls the locality)

# Create a range of query points for prediction
X_query = np.linspace(0, 10, 1000).reshape(-1, 1)

# Predict using Locally Weighted Regression for each query point
y_pred = np.array([locally_weighted_regression(X, y, tau, xq) for xq in X_query])

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_query, y_pred, color='red', label='Locally Weighted Regression (LWR)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression (LWR)')
plt.legend()
plt.show()
