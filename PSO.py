# PSO.py
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PSOClustering:
    def __init__(self, n_clusters=3, n_particles=30, max_iter=100, data=None, inertia_weight=0.7):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.data = data
        self.dim = n_clusters * data.shape[1]
        self.inertia_weight = inertia_weight
        self.cognitive = 1.5
        self.social = 1.5
        self.particles = None
        self.velocities = None
        self.p_best = None
        self.p_best_scores = None
        self.g_best = None
        self.g_best_score = None
        self.best_centroids = None
        self.labels_ = None

    def _init_particles(self):
        particles = np.random.rand(self.n_particles, self.dim)
        particles *= (np.max(self.data, axis=0).repeat(self.n_clusters) - np.min(self.data, axis=0).repeat(self.n_clusters))
        particles += np.min(self.data, axis=0).repeat(self.n_clusters)
        velocities = np.random.rand(self.n_particles, self.dim) - 0.5
        return particles, velocities

    def _evaluate(self, centroids):
        centroids = centroids.reshape(self.n_clusters, -1)
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        if len(np.unique(labels)) < 2:
            return -1
        score = silhouette_score(self.data, labels)
        return score

    def fit(self):
        X = self.data
        self.particles, self.velocities = self._init_particles()
        self.p_best = self.particles.copy()
        self.p_best_scores = np.array([self._evaluate(p) for p in self.particles])
        self.g_best = self.p_best[np.argmax(self.p_best_scores)]
        self.g_best_score = np.max(self.p_best_scores)

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.cognitive * r1 * (self.p_best[i] - self.particles[i]) +
                                      self.social * r2 * (self.g_best - self.particles[i]))
                self.particles[i] += self.velocities[i]

                score = self._evaluate(self.particles[i])
                if score > self.p_best_scores[i]:
                    self.p_best[i] = self.particles[i]
                    self.p_best_scores[i] = score
                    if score > self.g_best_score:
                        self.g_best = self.particles[i]
                        self.g_best_score = score

        self.best_centroids = self.g_best.reshape(self.n_clusters, -1)
        self.labels_ = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.best_centroids, axis=2), axis=1)
        return self

    def plot_clusters(self, X_vis):
        plt.figure(figsize=(10, 8))
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=self.labels_, cmap='viridis', alpha=0.6)
        # Transform centroids to PCA space for visualization
        pca = PCA(n_components=2)
        pca.fit(self.data)
        centroids_vis = pca.transform(self.best_centroids)
        plt.scatter(centroids_vis[:, 0], centroids_vis[:, 1], marker='X', color='red', s=200, linewidths=3)
        plt.title("Employee Clusters by PSO (PCA Reduced)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Cluster')
        plt.show()