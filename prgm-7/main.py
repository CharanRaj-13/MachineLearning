'''
pip install numpy pandas scikit-learn matplotlib seaborn

'''

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target
features = iris.feature_names

# Convert to DataFrame for easier handling
df = pd.DataFrame(data, columns=features)

# Optionally, scale the data (for better performance of K-Means)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply KMeans clustering (let's say we want to find 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Plot the first two features
sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='viridis', s=100)

# Mark the centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

# Titles and labels
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend()
plt.show()

# Output the centroids
print("Centroids of the clusters:", centroids)
