import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, epsilon=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=200, alpha=0.75)
        plt.show()
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.linalg.norm(new_centroids - self.centroids) < self.epsilon:
                break
            self.centroids = new_centroids
        return labels

def test_kmeans():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.75)
    plt.show()

if __name__ == "__main__":
    test_kmeans()