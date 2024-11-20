'''
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load a dataset (For example, the Boston Housing dataset)
# Using sklearn's dataset loading method
from sklearn.datasets import fetch_california_housing

# Load the Boston dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Features
y = data.target  # Target variable (housing prices)

# 2. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Linear Regression model
model = LinearRegression()

# 4. Train the model on the training data
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 7. Plot the actual vs predicted values for visualization
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# 8. Make predictions for new data
new_data = np.array([[0.1, 0.2, 7.2, 0.0, 0.4, 5.0, 4.0, 2.5, 1.0, 300, 18, 10.0, 300]])  # Example new data
predicted_price = model.predict(new_data)
print(f"Predicted housing price: {predicted_price[0]}")
