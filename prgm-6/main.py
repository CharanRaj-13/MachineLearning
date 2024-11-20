'''
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn

'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate sample data
np.random.seed(0)
data = np.random.rand(8, 5)

# Step 2: Normalize the data for cosine similarity
data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

# Step 3: Define the number of clusters
k = 3

# Step 4: Initialize centroids randomly
# Randomly select k data points as initial centroids
centroids = data_normalized[np.random.choice(data_normalized.shape[0], k, replace=False)]

# K-Means Algorithm with Cosine Similarity
max_iters = 100
for i in range(max_iters):
    # Compute cosine similarity between each point and each centroid
    similarity_matrix = cosine_similarity(data_normalized, centroids)
    
    # Assign each point to the cluster with the highest similarity
    clusters = np.argmax(similarity_matrix, axis=1)
    
    # Update centroids as the mean of points in each cluster
    new_centroids = np.array([data_normalized[clusters == j].mean(axis=0) for j in range(k)])
    
    # Check for convergence (if centroids do not change)
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

print("Cluster assignments:", clusters + 1)  # Adding 1 to match 1-indexed clusters

# Step 5: Visualize clusters
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=clusters, palette="viridis", s=100)
plt.title("K-Means Clustering Visualization with Cosine Similarity")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
