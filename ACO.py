# ACO.py
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class AntColonyClustering:
    def __init__(self, data, n_clusters=3, n_ants=10, n_iterations=50,
                 alpha=1.0, beta=2.0, evaporation_rate=0.5):
        self.data = data
        self.n_clusters = n_clusters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        # Initialize pheromone matrix
        self.pheromone = np.ones((len(data), n_clusters)) * 0.1
        self.best_solution = None
        self.best_centroids = None
        self.best_score = -np.inf
        # Initialize heuristic information (inverse distance to initial centroids)
        self.heuristic = self._initialize_heuristic()

    def _initialize_heuristic(self):
        # Use K-Means to initialize centroids for heuristic
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(self.data)
        centroids = kmeans.cluster_centers_
        # Compute inverse distances as heuristic
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        heuristic = 1 / (distances + 1e-6)  # Avoid division by zero
        return heuristic

    def _evaluate(self, labels):
        if len(np.unique(labels)) < 2:
            return -1.0
        return silhouette_score(self.data, labels)

    def _construct_solution(self):
        labels = np.zeros(len(self.data), dtype=int)
        for i in range(len(self.data)):
            # Combine pheromone and heuristic
            pher = self.pheromone[i] ** self.alpha
            heur = self.heuristic[i] ** self.beta
            prob = pher * heur
            prob = np.clip(prob, 1e-6, None)
            prob /= np.sum(prob)
            labels[i] = np.random.choice(self.n_clusters, p=prob)
        return labels

    def fit(self):
        for _ in range(self.n_iterations):
            all_labels = []
            all_scores = []
            for _ in range(self.n_ants):
                labels = self._construct_solution()
                score = self._evaluate(labels)
                all_labels.append(labels)
                all_scores.append(score)
                if score > self.best_score:
                    self.best_score = score
                    self.best_solution = labels.copy()
                    # Compute centroids for the best solution
                    self.best_centroids = np.array([np.mean(self.data[labels == k], axis=0)
                                                    for k in range(self.n_clusters)])

            # Update pheromones
            self.pheromone *= (1 - self.evaporation_rate)
            for labels, score in zip(all_labels, all_scores):
                if score <= 0:
                    continue
                for i, label in enumerate(labels):
                    self.pheromone[i, label] += score / self.n_ants

            self.pheromone = np.clip(self.pheromone, 1e-6, None)

        return self

    def predict(self):
        return self.best_solution