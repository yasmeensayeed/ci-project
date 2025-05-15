# gui.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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

# Optimal default parameters for each algorithm
def get_algorithms():
    return {
        'Genetic Algorithm': {
            'class': GeneticClustering,
            'params': {
                'pop_size': 50,
                'generations': 100,
                'mutation_rate': 0.15
            }
        },
        'PSO': {
            'class': PSOClustering,
            'params': {
                'n_particles': 40,
                'max_iter': 80,
                'inertia_weight': 0.729,
                'cognitive': 1.49445,
                'social': 1.49445
            }
        },
        'Ant Colony Optimization': {
            'class': AntColonyClustering,
            'params': {
                'n_ants': 30,
                'n_iterations': 70,
                'alpha': 1.2,
                'beta': 2.0,
                'evaporation_rate': 0.4
            }
        },
        'Hybrid (GA + PSO)': {
            'class': HybridClustering,
            'params': {
                'pop_size_ga': 30,
                'ga_generations': 60,
                'pso_particles': 40,
                'pso_iterations': 60,
                'mutation_rate': 0.1
            }
        },
        'K-Means': {
            'class': KMeans,
            'params': {
                'random_state': 42,
                'n_init': 15,
                'max_iter': 300
            }
        },
        'ACO Variation 1 (Random Centroids)': {
            'class': AntColonyClusteringV1,
            'params': {
                'n_ants': 30,
                'n_iterations': 70,
                'alpha': 1.2,
                'beta': 2.0,
                'evaporation_rate': 0.4
            }
        },
        'ACO Variation 2 (Davies-Bouldin)': {
            'class': AntColonyClusteringV2,
            'params': {
                'n_ants': 30,
                'n_iterations': 70,
                'alpha': 1.2,
                'beta': 2.0,
                'evaporation_rate': 0.4
            }
        },
        'ACO Variation 3 (Exploitation)': {
            'class': AntColonyClusteringV3,
            'params': {
                'n_ants': 30,
                'n_iterations': 70,
                'alpha': 1.2,
                'beta': 2.0,
                'evaporation_rate': 0.4,
                'q0': 0.9
            }
        },
        'ACO Variation 4 (Density-Based)': {
            'class': AntColonyClusteringV4,
            'params': {
                'n_ants': 30,
                'n_iterations': 70,
                'alpha': 1.2,
                'beta': 2.0,
                'evaporation_rate': 0.4
            }
        },
        'ACO Variation 5 (Dynamic Evaporation)': {
            'class': AntColonyClusteringV5,
            'params': {
                'n_ants': 30,
                'n_iterations': 70,
                'alpha': 1.2,
                'beta': 2.0,
                'evaporation_rate': 0.4,
                'pheromone_init': 0.1
            }
        },
    }

def get_cluster_names(labels, features_df, n_clusters):
    """Generate meaningful cluster names based on customer characteristics"""
    features_df['Cluster'] = labels
    cluster_names = []
    
    for cluster_num in range(n_clusters):
        cluster_data = features_df[features_df['Cluster'] == cluster_num]
        
        # Determine age group
        avg_age = cluster_data['Age'].mean()
        if avg_age < 30:
            age_group = 'Young'
        elif avg_age < 50:
            age_group = 'Middle-Aged'
        else:
            age_group = 'Senior'
        
        # Determine income level
        avg_income = cluster_data['Annual Income (k$)'].mean()
        if avg_income < 50:  # Assuming income is in thousands
            income_group = 'Low-Income'
        elif avg_income < 80:
            income_group = 'Mid-Income'
        else:
            income_group = 'High-Income'
        
        # Determine spending behavior
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        if avg_spending < 40:
            spending_group = 'Low-Spender'
        elif avg_spending < 60:
            spending_group = 'Moderate-Spender'
        else:
            spending_group = 'High-Spender'
        
        # Create composite name
        cluster_names.append(f"{age_group} {income_group} {spending_group}")
    
    return cluster_names

def plot_enhanced_clusters(X_vis, labels, centroids, algorithm_name, n_clusters, features_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get meaningful cluster names
    cluster_names = get_cluster_names(labels, features_df.copy(), n_clusters)
    
    # Create custom color map with distinct colors
    colors = plt.cm.get_cmap('viridis', n_clusters)
    
    # Plot data points with cluster colors
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], 
                        c=labels, cmap=colors, alpha=0.7, 
                        edgecolors='w', linewidth=0.5)
    
    # Plot centroids if available
    if centroids is not None and centroids.ndim == 2:
        pca = PCA(n_components=2)
        pca.fit(X)
        centroids_vis = pca.transform(centroids)
        ax.scatter(centroids_vis[:, 0], centroids_vis[:, 1],
                  c='red', marker='X', s=300, linewidths=3,
                  edgecolors='black', label='Centroids')
    
    # Add cluster information with meaningful names
    for cluster_num in range(n_clusters):
        cluster_points = X_vis[labels == cluster_num]
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            ax.text(center[0], center[1], cluster_names[cluster_num],
                   fontsize=10, weight='bold', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xlabel("Age, Income, and Spending Score (PCA 1)", fontsize=10)
    ax.set_ylabel("Age, Income, and Spending Score (PCA 2)", fontsize=10)
    ax.set_title(f"{algorithm_name} Customer Clustering Results\n({n_clusters} Clusters)", fontsize=14, pad=20)
    
    # Enhanced colorbar with meaningful names
    cbar = plt.colorbar(scatter, ticks=range(n_clusters))
    cbar.set_label('Customer Segments', rotation=270, labelpad=20)
    cbar.set_ticklabels(cluster_names)
    
    if centroids is not None:
        ax.legend(loc='upper right')
    
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

# Main GUI code
st.title("Customer Segmentation with CI/EC Algorithms")

# Data loading
X, X_vis, features_df = load_data2(file_path='Mall_Customers.csv')
if X is None:
    st.error("Please make sure Mall_Customers.csv is in the directory.")
    st.stop()

# Algorithm selection
algos = get_algorithms()
choice = st.selectbox("Select algorithm", list(algos.keys()))
config = algos[choice]

# Parameter tuning UI
st.sidebar.subheader(f"Parameters for {choice}")
user_params = {}
for param, default in config['params'].items():
    if param == 'random_state':
        user_params[param] = default
    elif isinstance(default, int):
        user_params[param] = st.sidebar.slider(
            param, 
            min_value=1,
            max_value=500 if 'size' in param.lower() else 200,
            value=default
        )
    else:
        user_params[param] = st.sidebar.slider(
            param,
            min_value=0.0,
            max_value=2.0 if param in ['alpha', 'beta'] else 1.0,
            value=default,
            step=0.001 if 'rate' in param.lower() else 0.01
        )

# Number of clusters
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)

if st.button("Run Clustering"):
    AlgorithmClass = config['class']
    params = {'n_clusters': n_clusters}
    for param in config['params']:
        if param in user_params:
            params[param] = user_params[param]
    if choice != 'K-Means':
        params['data'] = X
    
    model = AlgorithmClass(**params)
    
    # Fit/evolve the model
    if hasattr(model, 'evolve'):
        centroids = model.evolve()
        labels = model.assign_clusters(centroids)
    elif choice == 'K-Means':
        model.fit(X)
        labels = model.labels_
        centroids = model.cluster_centers_
    else:
        model.fit()
        labels = model.predict() if hasattr(model, 'predict') else model.labels_
        centroid_attrs = ['centroids', 'best_centroids', 'best_solution']
        centroids = None
        for attr in centroid_attrs:
            if hasattr(model, attr):
                centroids = getattr(model, attr)
                break
        if centroids is not None and centroids.ndim == 1:
            labels = centroids
            centroids = np.array([np.mean(X[labels == k], axis=0) for k in range(n_clusters)])

    # Evaluation
    try:
        score = silhouette_score(X, labels)
        st.success(f"Silhouette Score: {score:.4f}")
        if score > 0.5:
            st.info("Good cluster separation achieved!")
        elif score > 0.3:
            st.warning("Moderate cluster separation")
        else:
            st.error("Poor cluster separation - try adjusting parameters")
    except:
        st.error("Could not compute silhouette score - invalid clustering")

    # Enhanced plotting with meaningful cluster names
    plot_enhanced_clusters(X_vis, labels, centroids, choice, n_clusters, features_df)
    
    # Cluster characteristics with meaningful names (optional)
    if st.checkbox("Show detailed cluster characteristics"):
        features_df['Cluster'] = labels
        cluster_names = get_cluster_names(labels, features_df.copy(), n_clusters)
        features_df['Cluster_Name'] = features_df['Cluster'].map(lambda x: cluster_names[x])
        
        st.subheader("Cluster Profiles")
        cluster_stats = features_df.groupby('Cluster_Name').agg({
            'Age': ['mean', 'std'],
            'Annual_Income': ['mean', 'std'],
            'Spending_Score': ['mean', 'std']
        })
        
        st.dataframe(cluster_stats.style.background_gradient(cmap='viridis'))