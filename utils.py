# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# def load_data(file_path='MFG10YearTerminationData.csv', sample_size=700):
#     """
#     Optimized feature selection based on correlation analysis:
#     Selected Features:
#     1. age (strong demographic feature)
#     2. length_of_service (employment duration)
#     3. termination_category (engineered feature)
#     4. service_ratio (new engineered feature)
#     """
#     try:
#         # Load data with proper null handling
#         data = pd.read_csv(file_path)
        
#         # 1. Feature Engineering
#         # Create termination categories
#         data['termination_category'] = data['termreason_desc'].apply(
#             lambda x: 3 if 'retire' in str(x).lower() else       # Retirement
#                      2 if 'resign' in str(x).lower() else        # Resignation
#                      1 if 'term' in str(x).lower() else          # Termination
#                      0 if pd.notnull(x) else                     # Other
#                      -1)                                         # Active
        
#         # Create service ratio (age vs length of service)
#         data['service_ratio'] = data['length_of_service'] / (data['age'] - 18 + 1e-6)
        
#         # 2. Select optimal features (based on correlation analysis)
#         features = [
#             'age',
#             'length_of_service',
#             'service_ratio',
#             'termination_category'
#         ]
        
#         # 3. Handle missing values
#         imputer = SimpleImputer(strategy='median')
#         data[features] = imputer.fit_transform(data[features])
        
#         # 4. Remove outliers (3-sigma rule)
#         for feature in ['age', 'length_of_service', 'service_ratio']:
#             mean = data[feature].mean()
#             std = data[feature].std()
#             data = data[(data[feature] > mean - 3*std) & 
#                        (data[feature] < mean + 3*std)]
        
#         # 5. Balanced sampling
#         terminated = data[data['STATUS'] == 'TERMINATED']
#         active = data[data['STATUS'] == 'ACTIVE']
#         sample_size = min(sample_size, len(data))
#         terminated_sample = terminated.sample(min(len(terminated), sample_size//2))
#         active_sample = active.sample(min(len(active), sample_size - len(terminated_sample)))
#         data = pd.concat([terminated_sample, active_sample])
        
#         # 6. Feature scaling
#         scaler = StandardScaler()
#         X = scaler.fit_transform(data[features])
        
#         # 7. Dimensionality reduction for visualization
#         pca = PCA(n_components=2)
#         X_vis = pca.fit_transform(X)
        
#         return X, X_vis, data[features]
    
#     except Exception as e:
#         print(f"Error loading data: {str(e)}")
#         return None, None, None

def load_data2(file_path='Mall_Customers.csv', sample_size=700):
    """
    Load and preprocess Mall_Customers dataset for clustering.
    Selected Features:
    1. Age (customer age)
    2. Annual_Income (annual income in thousands)
    3. Spending_Score (spending score from 1-100)
    """
    try:
        # Load data with proper null handling
        data = pd.read_csv(file_path)
        
        # Select features
        features = [
            'Age',
            'Annual Income (k$)',
            'Spending Score (1-100)'
        ]
        
        # Verify all required columns exist
        missing_cols = [col for col in features if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        data[features] = imputer.fit_transform(data[features])
        
        # Remove outliers (3-sigma rule)
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            data = data[(data[feature] > mean - 3*std) & 
                       (data[feature] < mean + 3*std)]
        
        # Random sampling if sample_size is specified
        if sample_size < len(data):
            data = data.sample(n=sample_size, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(data[features])
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        
        return X, X_vis, data[features]
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

def plot_clusters(X_vis, labels, centers=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='viridis', alpha=0.6)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.title("Optimized Employee Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def evaluate_clustering(X, labels):
    try:
        score = silhouette_score(X, labels)
        print(f"Silhouette Score: {score:.4f}")
        if score > 0.5:
            print("Good cluster separation achieved!")
        elif score > 0.3:
            print("Moderate cluster separation")
        else:
            print("Poor cluster separation - consider feature engineering")
        return score
    except:
        print("Invalid clustering (e.g., single cluster).")
        return -1.0