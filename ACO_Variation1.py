import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class AntColonyClusteringV1:
    def __init__(self, data, n_clusters=3, n_ants=10, n_iterations=50,
                 alpha=1.0, beta=2.0, evaporation_rate=0.5):
        self.data = data
        self.n_clusters = n_clusters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones((len(data), n_clusters)) * 0.1
        self.best_solution = None
        self.best_centroids = None
        self.best_score = -np.inf
        self.heuristic = self._initialize_heuristic()

    def _initialize_heuristic(self):
        # Random centroid initialization instead of K-Means
        indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        centroids = self.data[indices]
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        heuristic = 1 / (distances + 1e-6)
        return heuristic

    def _evaluate(self, labels):
        if len(np.unique(labels)) < 2:
            return -1.0
        return silhouette_score(self.data, labels)

    def _construct_solution(self):
        labels = np.zeros(len(self.data), dtype=int)
        for i in range(len(self.data)):
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
                    self.best_centroids = np.array([np.mean(self.data[labels == k], axis=0)
                                                    for k in range(self.n_clusters)])

            # Elitist pheromone update: only best ant contributes
            self.pheromone *= (1 - self.evaporation_rate)
            best_idx = np.argmax(all_scores)
            best_labels = all_labels[best_idx]
            best_score = all_scores[best_idx]
            if best_score > 0:
                for i, label in enumerate(best_labels):
                    self.pheromone[i, label] += best_score

            self.pheromone = np.clip(self.pheromone, 1e-6, None)

        return self

    def predict(self):
        return self.best_solution