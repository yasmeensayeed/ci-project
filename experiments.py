# experiments.py
import numpy as np
import random
import pandas as pd
import contextlib
import io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import product
from utils import load_data2
from GeneticClustering import GeneticClustering
from PSO import PSOClustering
from ACO import AntColonyClustering
from hybrid import HybridClustering
from ACO_Variation1 import AntColonyClusteringV1
from ACO_Variation2 import AntColonyClusteringV2
from ACO_Variation3 import AntColonyClusteringV3
from ACO_Variation4 import AntColonyClusteringV4
from ACO_Variation5 import AntColonyClusteringV5

# Parameter grids for sensitivity analysis
param_grids = {
    'GA': {
        'mutation_rate': [0.1, 0.2, 0.3],
        'pop_size': [10, 20, 30]
    },
    'PSO': {
        'inertia_weight': [0.5, 0.7, 0.9],
        'n_particles': [20, 30, 40]
    },

    'ACO': {
        'evaporation_rate': [0.3, 0.5, 0.7],
        'n_ants': [10, 20, 30]
    },

    'Hybrid': {
        'mutation_rate': [0.1, 0.2, 0.3],
        'pop_size_ga': [10, 20, 30]
    },
    
    'ACO_V1': {
        'evaporation_rate': [0.3, 0.5, 0.7],
        'n_ants': [10, 20, 30]
    },
    'ACO_V2': {
        'evaporation_rate': [0.3, 0.5, 0.7],
        'n_ants': [10, 20, 30]
    },
    'ACO_V3': {
        'evaporation_rate':	[0.3, 0.5, 0.7],
        'n_ants': [10, 20, 30],
        'q0': [0.8, 0.9, 0.95]
    },
    'ACO_V4': {
        'evaporation_rate': [0.3, 0.5, 0.7],
        'n_ants': [10, 20, 30]
    },
    'ACO_V5': {
        'evaporation_rate': [0.3, 0.5, 0.7],
        'n_ants': [10, 20, 30],
        'pheromone_init': [0.05, 0.1, 0.2]
    }
}

# Algorithm configurations
algorithms = {
    'GA': {
        'class': GeneticClustering,
        'params': {
            'data': None,
            'n_clusters': 3,
            'pop_size': 20,
            'generations': 50,
            'mutation_rate': 0.2
        },
        'suppress_output': True
    },
    'PSO': {
        'class': PSOClustering,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_particles': 30,
            'max_iter': 50,
            'inertia_weight': 0.7
        },
        'suppress_output': False
    },
    'ACO': {
        'class': AntColonyClustering,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_ants': 20,
            'n_iterations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5
        },
        'suppress_output': False
    },
    'Hybrid': {
        'class': HybridClustering,
        'params': {
            'data': None,
            'n_clusters': 3,
            'pop_size_ga': 20,
            'ga_generations': 50,
            'pso_particles': 30,
            'pso_iterations': 50,
            'mutation_rate': 0.2
        },
        'suppress_output': False
    },
    'KMeans': {
        'class': KMeans,
        'params': {
            'n_clusters': 3,
            'random_state': 42
        },
        'suppress_output': False
    },
    'ACO_V1': {
        'class': AntColonyClusteringV1,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_ants': 20,
            'n_iterations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5
        },
        'suppress_output': False
    },
    'ACO_V2': {
        'class': AntColonyClusteringV2,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_ants': 20,
            'n_iterations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5
        },
        'suppress_output': False
    },
    'ACO_V3': {
        'class': AntColonyClusteringV3,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_ants': 20,
            'n_iterations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5,
            'q0': 0.9
        },
        'suppress_output': False
    },
    'ACO_V4': {
        'class': AntColonyClusteringV4,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_ants': 20,
            'n_iterations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5
        },
        'suppress_output': False
    },
    'ACO_V5': {
        'class': AntColonyClusteringV5,
        'params': {
            'data': None,
            'n_clusters': 3,
            'n_ants': 20,
            'n_iterations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.5,
            'pheromone_init': 0.1
        },
        'suppress_output': False
    },
    
}
# Seeds for reproducibility
seeds = list(range(30))

# At the start of experiments, save seeds:
with open('seeds.txt', 'w') as f:
    f.write('\n'.join(map(str, seeds)) + '\n')

# Load dataset
X, _ , _ = load_data2()  
if X is None:
    exit(1)

# Run experiments
results = []
for name, algo in algorithms.items():
    print(f"Starting experiments for {name}...")
    param_grid = param_grids.get(name, {})
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())] if param_grid else [{}]
    
    for param_set in param_combinations:
        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)
            params = algo['params'].copy()
            params.update(param_set)
            if name != 'KMeans':
                params['data'] = X
            model = algo['class'](**params)

            if name == 'GA' and algo.get('suppress_output', False):
                with contextlib.redirect_stdout(io.StringIO()):
                    centroids = model.evolve()
                labels = model.assign_clusters(centroids)
            elif name == 'KMeans':
                model.fit(X)
                labels = model.labels_
            else:
                model.fit()
                labels = model.predict() if hasattr(model, 'predict') else model.labels_

            try:
                score = silhouette_score(X, labels)
            except ValueError:
                score = -1.0
            result = {'algorithm': name, 'seed': seed, 'silhouette': score}
            result.update(param_set)
            results.append(result)

        # Save intermediate results
        pd.DataFrame(results).to_csv('experiments_results.csv', index=False)
        print(f"Completed {name} with params {param_set}. Intermediate results saved.")

print('All experiments done. Final results are saved in experiments_results.csv')