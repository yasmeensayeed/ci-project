# GeneticClustering.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import random

class GeneticClustering:
    def __init__(self, data, n_clusters=4, pop_size=20, generations=50, mutation_rate=0.2):
        self.data = data
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n_features = data.shape[1]
        self.chromosome_length = n_clusters * self.n_features
        if self.chromosome_length <= 0:
            raise ValueError("Invalid chromosome length. Check n_clusters and data dimensions.")

    def initialize_population(self):
        return [np.random.uniform(np.min(self.data), np.max(self.data), self.chromosome_length)
                for _ in range(self.pop_size)]

    def decode_chromosome(self, chromosome):
        if chromosome.size != self.chromosome_length:
            raise ValueError(f"Chromosome size {chromosome.size} does not match expected {self.chromosome_length}")
        return chromosome.reshape((self.n_clusters, self.n_features))

    def assign_clusters(self, centroids):
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fitness(self, chromosome):
        try:
            centroids = self.decode_chromosome(chromosome)
            labels = self.assign_clusters(centroids)
            score = silhouette_score(self.data, labels)
        except:
            score = -1
        return score

    def select_parents(self, population, fitnesses):
        # Tournament selection
        tournament_size = 3
        selected = []
        for _ in range(2):  # Select 2 parents
            contestants = np.random.choice(len(population), tournament_size, replace=False)
            contestant_fitness = [fitnesses[i] for i in contestants]
            winner = contestants[np.argmax(contestant_fitness)]
            selected.append(population[winner])
        return selected

    def crossover(self, parent1, parent2):
        # Uniform crossover
        mask = np.random.rand(self.chromosome_length) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(self, chromosome):
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 0.5)
        return np.clip(chromosome, np.min(self.data), np.max(self.data))

    def evolve(self):
        population = self.initialize_population()
        best_chromosome = None
        best_fitness = -1

        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            new_population = []

            for _ in range(self.pop_size):
                parents = self.select_parents(population, fitnesses)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                new_population.append(child)

            population = new_population
            gen_best_fit = max(fitnesses)
            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_chromosome = population[np.argmax(fitnesses)]

            print(f"Generation {gen + 1}, Best Fitness: {best_fitness:.4f}")

        return self.decode_chromosome(best_chromosome)