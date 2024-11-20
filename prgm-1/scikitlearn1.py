'''
pip install scikit-learn
pip install numpy

'''

from sklearn.linear_model import LinearRegression
import numpy as np

# Example: Simple linear regression
X = np.array([[1], [2], [3], [4]])
y = np.array([4, 3, 2, 1])
model = LinearRegression().fit(X, y)
prediction = model.predict(np.array([[5]]))
print("Prediction for 5:", prediction)

# Used for implementing machine learning algorithms (e.g., regression, classification, clustering) and model evaluation.