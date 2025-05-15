# hybrid.py
import numpy as np
import contextlib
import io
from sklearn.metrics import silhouette_score
from GeneticClustering import GeneticClustering
from PSO import PSOClustering

class HybridClustering:
    """
    Competitive coevolutionary hybrid of GA and PSO.
    Reference: Shi et al. (2005) on hybrid evolutionary algorithms.
    """
    def __init__(self, data, n_clusters=3,
                 pop_size_ga=20, ga_generations=50,
                 pso_particles=30, pso_iterations=50,
                 mutation_rate=0.2):
        self.data = data
        self.n_clusters = n_clusters
        self.pop_size_ga = pop_size_ga
        self.ga_generations = ga_generations
        self.pso_particles = pso_particles
        self.pso_iterations = pso_iterations
        self.mutation_rate = mutation_rate
        self.n_features = data.shape[1]
        self.centroids = None
        self.labels_ = None
        self.best_score = -np.inf

    def _evaluate(self, centroids):
        centroids = centroids.reshape(self.n_clusters, self.n_features)
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        try:
            return silhouette_score(self.data, labels)
        except:
            return -1.0

    def fit(self):
        # Initialize GA and PSO
        ga = GeneticClustering(
            data=self.data,
            n_clusters=self.n_clusters,
            pop_size=self.pop_size_ga,
            generations=1,  # Run one generation per iteration
            mutation_rate=self.mutation_rate
        )
        pso = PSOClustering(
            n_clusters=self.n_clusters,
            n_particles=self.pso_particles,
            max_iter=1,  # Run one iteration per cycle
            data=self.data
        )

        # Initialize populations
        with contextlib.redirect_stdout(io.StringIO()):
            ga_population = ga.initialize_population()
            pso_particles, pso_velocities = pso._init_particles()

        ga_scores = np.array([ga.fitness(ind) for ind in ga_population])
        pso_scores = np.array([pso._evaluate(p) for p in pso_particles])
        best_individual = ga_population[np.argmax(ga_scores)] if max(ga_scores) > max(pso_scores) else pso_particles[np.argmax(pso_scores)]
        self.best_score = max(max(ga_scores), max(pso_scores))

        # Competitive coevolution loop
        for _ in range(max(self.ga_generations, self.pso_iterations)):
            # GA step
            new_population = []
            with contextlib.redirect_stdout(io.StringIO()):
                fitnesses = [ga.fitness(ind) for ind in ga_population]
                for _ in range(self.pop_size_ga):
                    parents = ga.select_parents(ga_population, fitnesses)
                    child = ga.crossover(parents[0], parents[1])
                    child = ga.mutate(child)
                    new_population.append(child)
                ga_population = new_population
            ga_scores = np.array([ga.fitness(ind) for ind in ga_population])

            # PSO step
            with contextlib.redirect_stdout(io.StringIO()):
                pso.fit()  # One iteration
            pso_particles = pso.particles if hasattr(pso, 'particles') else pso._init_particles()[0]
            pso_scores = np.array([pso._evaluate(p) for p in pso_particles])

            # Competition: Replace worst individuals with best from other population
            ga_worst_idx = np.argmin(ga_scores)
            pso_best_idx = np.argmax(pso_scores)
            if pso_scores[pso_best_idx] > ga_scores[ga_worst_idx]:
                ga_population[ga_worst_idx] = pso_particles[pso_best_idx].copy()

            pso_worst_idx = np.argmin(pso_scores)
            ga_best_idx = np.argmax(ga_scores)
            if ga_scores[ga_best_idx] > pso_scores[pso_worst_idx]:
                pso_particles[pso_worst_idx] = ga_population[ga_best_idx].copy()

            # Update best solution
            current_best_score = max(max(ga_scores), max(pso_scores))
            if current_best_score > self.best_score:
                self.best_score = current_best_score
                best_individual = ga_population[np.argmax(ga_scores)] if max(ga_scores) > max(pso_scores) else pso_particles[np.argmax(pso_scores)]

        # Finalize
        self.centroids = best_individual.reshape(self.n_clusters, self.n_features)
        self.labels_ = np.argmin(np.linalg.norm(self.data[:, np.newaxis] - self.centroids, axis=2), axis=1)
        return self.centroids

    def predict(self, X=None):
        data = self.data if X is None else X
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)